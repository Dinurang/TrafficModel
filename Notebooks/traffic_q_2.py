# -*- coding: utf-8 -*-
"""traffic_q_learning_enhanced.ipynb

# Q-Learning for Dynamic 4-Way Traffic Light Control with Real-time Data Collection
"""

import numpy as np
import random
import pickle
import time
from collections import defaultdict
from datetime import datetime
import os

# Configuration
NUM_ROADS = 4
GREEN_TIME_OPTIONS = [15, 30, 45, 60, 75, 90, 105, 120]  # Extended up to 120 seconds
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2
MODEL_FILE = "traffic_q_model.pkl"
HISTORY_FILE = "traffic_data_history.pkl"

def discretize_flow(flow):
    """Convert continuous flow values to discrete categories"""
    if flow < 5:
        return 0  # Very low
    elif flow < 15:
        return 1  # Low
    elif flow < 30:
        return 2  # Medium
    elif flow < 50:
        return 3  # High
    else:
        return 4  # Very high

def get_state(road_priority, day, time_slot, flows):
    """Create a state representation from inputs"""
    discretized_flows = tuple(discretize_flow(f) for f in flows)
    return (road_priority, day, time_slot, discretized_flows)

# Generate all possible green time combinations
ACTIONS = []
for g1 in GREEN_TIME_OPTIONS:
    for g2 in GREEN_TIME_OPTIONS:
        for g3 in GREEN_TIME_OPTIONS:
            for g4 in GREEN_TIME_OPTIONS:
                # Ensure minimum total cycle time constraint
                total_time = g1 + g2 + g3 + g4
                if 60 <= total_time <= 180:  # Reasonable cycle time constraints
                    ACTIONS.append((g1, g2, g3, g4))

NUM_ACTIONS = len(ACTIONS)
print(f"Total actions: {NUM_ACTIONS} (Filtered for reasonable cycle times)")

def compute_reward(flows, green_times, road_priority, previous_reward=None):
    """Compute reward for given traffic conditions"""
    incoming = flows[:4]
    outgoing = flows[4:]
    
    # Calculate congestion
    congestion = 0
    waiting_times = []
    for i in range(4):
        excess = max(0, incoming[i] - outgoing[i])
        congestion += excess
        # Estimate waiting time based on queue length
        if incoming[i] > 0:
            wait = excess / max(1, outgoing[i]) * 10
            waiting_times.append(wait)
    
    # Priority bonus (more green for priority road)
    priority_bonus = green_times[road_priority] * 0.3
    
    # Efficiency penalty (avoid extreme green times)
    efficiency_penalty = 0
    for gt in green_times:
        if gt > 90:  # Penalize very long green times
            efficiency_penalty += (gt - 90) * 0.1
    
    # Smoothness penalty (avoid drastic changes)
    smoothness_penalty = 0
    if previous_reward is not None:
        time_diffs = [abs(green_times[i] - green_times[(i+1)%4]) for i in range(4)]
        smoothness_penalty = max(time_diffs) * 0.05
    
    avg_wait = np.mean(waiting_times) if waiting_times else 0
    reward = -congestion - avg_wait + priority_bonus - efficiency_penalty - smoothness_penalty
    
    return reward

class TrafficQLearning:
    def __init__(self):
        self.Q = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self.training_history = []
        self.load_model()
    
    def choose_action(self, state, epsilon=EPSILON):
        """Choose action using ε-greedy policy"""
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        return np.argmax(self.Q[state])
    
    def update_q_value(self, state, action_idx, reward, next_state):
        """Update Q-value using Q-learning formula"""
        best_next = np.max(self.Q[next_state]) if next_state in self.Q else 0
        current_q = self.Q[state][action_idx]
        self.Q[state][action_idx] = current_q + ALPHA * (reward + GAMMA * best_next - current_q)
    
    def save_model(self):
        """Save Q-table and training history"""
        model_data = {
            'Q': dict(self.Q),
            'training_history': self.training_history,
            'timestamp': datetime.now()
        }
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {MODEL_FILE}")
    
    def load_model(self):
        """Load Q-table from file if exists"""
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, 'rb') as f:
                    model_data = pickle.load(f)
                self.Q = defaultdict(lambda: np.zeros(NUM_ACTIONS))
                self.Q.update(model_data['Q'])
                self.training_history = model_data.get('training_history', [])
                print(f"Model loaded from {MODEL_FILE}")
                print(f"Loaded {len(self.Q)} states")
            except Exception as e:
                print(f"Error loading model: {e}. Starting fresh.")
    
    def train_with_data(self, data_point, episodes_per_data=10):
        """Train the model with a new data point"""
        road_priority, day, time_slot, flows = data_point
        state = get_state(road_priority, day, time_slot, flows)
        
        for _ in range(episodes_per_data):
            action_idx = self.choose_action(state, epsilon=0.3)  # Higher epsilon for exploration
            green_times = ACTIONS[action_idx]
            
            # Simulate next state (with small random changes to flows)
            next_flows = [max(0, f + random.randint(-5, 5)) for f in flows]
            next_state = get_state(road_priority, day, time_slot, next_flows)
            
            reward = compute_reward(flows, green_times, road_priority)
            self.update_q_value(state, action_idx, reward, next_state)
            
            # Record training history
            self.training_history.append({
                'timestamp': datetime.now(),
                'state': state,
                'action': green_times,
                'reward': reward,
                'flows': flows
            })
        
        # Save periodically
        if len(self.training_history) % 100 == 0:
            self.save_model()
    
    def predict_green_times(self, road_priority, day, time_slot, flows):
        """Predict optimal green times for given conditions"""
        state = get_state(road_priority, day, time_slot, flows)
        
        if state not in self.Q:
            # If state is unknown, use heuristic
            return self.heuristic_green_times(flows, road_priority)
        
        best_action_idx = np.argmax(self.Q[state])
        return ACTIONS[best_action_idx]
    
    def heuristic_green_times(self, flows, road_priority):
        """Heuristic fallback when Q-table doesn't have the state"""
        incoming = flows[:4]
        total_incoming = sum(incoming)
        
        if total_incoming == 0:
            return (30, 30, 30, 30)  # Default equal times
        
        # Allocate green times proportional to traffic, with bonus for priority road
        base_times = []
        for i in range(4):
            ratio = incoming[i] / total_incoming
            base_time = 30 + int(ratio * 60)  # Base 30-90 seconds
            base_times.append(min(120, max(15, base_time)))
        
        # Add priority bonus
        base_times[road_priority] = min(120, base_times[road_priority] + 15)
        
        # Normalize to reasonable cycle time
        total = sum(base_times)
        if total > 180:
            scale = 180 / total
            base_times = [int(t * scale) for t in base_times]
        
        return tuple(base_times)

def get_current_time_info():
    """Get current day and time slot"""
    now = datetime.now()
    day = now.weekday()  # 0=Monday, 6=Sunday
    hour = now.hour
    minute = now.minute
    
    # Convert to 15-minute time slots (0-95 per day)
    time_slot = hour * 4 + minute // 15
    
    return day, time_slot

def collect_manual_data(traffic_model):
    """Collect traffic flow data manually from user"""
    print("\n" + "="*50)
    print("TRAFFIC DATA COLLECTION")
    print("="*50)
    
    # Get road priority
    print("\nRoad Priority (which road should get more green time):")
    print("0: North")
    print("1: East")
    print("2: South")
    print("3: West")
    
    while True:
        try:
            road_priority = int(input("Enter road priority (0-3): "))
            if 0 <= road_priority <= 3:
                break
            else:
                print("Please enter a number between 0 and 3")
        except ValueError:
            print("Please enter a valid number")
    
    # Get current time or custom time
    use_current = input("Use current time? (y/n): ").lower()
    if use_current == 'y':
        day, time_slot = get_current_time_info()
        print(f"Using current time: Day={day}, Time Slot={time_slot}")
    else:
        while True:
            try:
                day = int(input("Enter day (0=Monday, 6=Sunday): "))
                if 0 <= day <= 6:
                    break
                else:
                    print("Please enter a number between 0 and 6")
            except ValueError:
                print("Please enter a valid number")
        
        while True:
            try:
                hour = int(input("Enter hour (0-23): "))
                if 0 <= hour <= 23:
                    break
                else:
                    print("Please enter a number between 0 and 23")
            except ValueError:
                print("Please enter a valid number")
        
        while True:
            try:
                minute = int(input("Enter minute (0-59): "))
                if 0 <= minute <= 59:
                    break
                else:
                    print("Please enter a number between 0 and 59")
            except ValueError:
                print("Please enter a valid number")
        
        time_slot = hour * 4 + minute // 15
    
    # Collect flow data
    print("\nEnter vehicle counts per 15 seconds:")
    directions = ["North", "East", "South", "West"]
    flows = []
    
    print("\nINCOMING traffic (vehicles entering intersection):")
    for i in range(4):
        while True:
            try:
                flow = int(input(f"{directions[i]} incoming: "))
                if flow >= 0:
                    flows.append(flow)
                    break
                else:
                    print("Please enter a non-negative number")
            except ValueError:
                print("Please enter a valid number")
    
    print("\nOUTGOING traffic (vehicles leaving intersection):")
    for i in range(4):
        while True:
            try:
                flow = int(input(f"{directions[i]} outgoing: "))
                if flow >= 0:
                    flows.append(flow)
                    break
                else:
                    print("Please enter a non-negative number")
            except ValueError:
                print("Please enter a valid number")
    
    # Create data point
    data_point = (road_priority, day, time_slot, flows)
    
    # Train with this data
    print("\nTraining model with new data...")
    traffic_model.train_with_data(data_point, episodes_per_data=20)
    
    # Get prediction
    green_times = traffic_model.predict_green_times(road_priority, day, time_slot, flows)
    
    print("\n" + "="*50)
    print("RECOMMENDED GREEN TIMES:")
    print("="*50)
    total_time = sum(green_times)
    for i in range(4):
        print(f"{directions[i]:6}: {green_times[i]:3} seconds ({green_times[i]/total_time*100:.1f}%)")
    print(f"Total cycle time: {total_time} seconds")
    
    return data_point

def batch_training_mode(traffic_model):
    """Batch training with multiple data points"""
    print("\n" + "="*50)
    print("BATCH TRAINING MODE")
    print("="*50)
    
    while True:
        try:
            n_points = int(input("How many data points to collect? "))
            if n_points > 0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    all_data = []
    for i in range(n_points):
        print(f"\nData point {i+1}/{n_points}:")
        
        # Get road priority
        while True:
            try:
                road_priority = int(input("Road priority (0-3): "))
                if 0 <= road_priority <= 3:
                    break
                else:
                    print("Please enter a number between 0 and 3")
            except ValueError:
                print("Please enter a valid number")
        
        # Get day
        while True:
            try:
                day = int(input("Day (0=Monday, 6=Sunday): "))
                if 0 <= day <= 6:
                    break
                else:
                    print("Please enter a number between 0 and 6")
            except ValueError:
                print("Please enter a valid number")
        
        # Get hour
        while True:
            try:
                hour = int(input("Hour (0-23): "))
                if 0 <= hour <= 23:
                    break
                else:
                    print("Please enter a number between 0 and 23")
            except ValueError:
                print("Please enter a valid number")
        
        # Get minute
        while True:
            try:
                minute = int(input("Minute (0-59): "))
                if 0 <= minute <= 59:
                    break
                else:
                    print("Please enter a number between 0 and 59")
            except ValueError:
                print("Please enter a valid number")
        
        time_slot = hour * 4 + minute // 15
        
        # Simplified flow input
        print("Enter 8 flow values (4 incoming, 4 outgoing) separated by commas:")
        while True:
            flow_input = input("Flows: ")
            try:
                flows = [int(x.strip()) for x in flow_input.split(',')]
                if len(flows) == 8 and all(f >= 0 for f in flows):
                    break
                elif len(flows) != 8:
                    print("Error: Need exactly 8 flow values")
                else:
                    print("Error: All values must be non-negative")
            except ValueError:
                print("Error: Please enter valid numbers separated by commas")
        
        data_point = (road_priority, day, time_slot, flows)
        all_data.append(data_point)
    
    # Train with all data
    print(f"\nTraining with {len(all_data)} data points...")
    for data_point in all_data:
        traffic_model.train_with_data(data_point, episodes_per_data=15)
    
    traffic_model.save_model()
    print("Batch training completed!")

def auto_collect_demo(traffic_model, duration_hours=24):
    """Auto-collect demo data for initial training"""
    print(f"\nGenerating demo data for {duration_hours} hours...")
    
    # Simulate a day of traffic patterns
    for hour in range(duration_hours):
        for quarter in range(4):  # 15-minute intervals
            time_slot = hour * 4 + quarter
            
            # Simulate different traffic patterns throughout the day
            if 7 <= hour <= 9:  # Morning rush
                base_flow = random.randint(30, 70)
                road_priority = 1  # East (main commute direction)
            elif 16 <= hour <= 18:  # Evening rush
                base_flow = random.randint(30, 70)
                road_priority = 3  # West (return commute)
            else:  # Off-peak
                base_flow = random.randint(10, 40)
                road_priority = random.randint(0, 3)
            
            # Generate flows with some randomness
            flows = []
            for _ in range(4):  # Incoming
                flows.append(max(0, base_flow + random.randint(-10, 10)))
            for _ in range(4):  # Outgoing
                flows.append(max(0, base_flow + random.randint(-10, 5)))
            
            # Use all days of week
            for day in range(7):
                data_point = (road_priority, day, time_slot, flows)
                traffic_model.train_with_data(data_point, episodes_per_data=2)
    
    traffic_model.save_model()
    print("Demo training completed!")

def main():
    """Main application loop"""
    traffic_model = TrafficQLearning()
    
    print("="*50)
    print("ADAPTIVE TRAFFIC LIGHT CONTROL SYSTEM")
    print("="*50)
    
    # Initial training option
    if len(traffic_model.Q) == 0:
        print("No existing model found.")
        train_demo = input("Train with demo data first? (y/n): ").lower()
        if train_demo == 'y':
            auto_collect_demo(traffic_model, duration_hours=12)
    
    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. Collect traffic data manually (15-second interval)")
        print("2. Batch training mode")
        print("3. Predict green times for current conditions")
        print("4. Show model statistics")
        print("5. Save model")
        print("6. Run demo simulation")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ")
        
        if choice == '1':
            collect_manual_data(traffic_model)
            
        elif choice == '2':
            batch_training_mode(traffic_model)
            
        elif choice == '3':
            # Predict for current conditions
            while True:
                try:
                    road_priority = int(input("Enter road priority (0-3): "))
                    if 0 <= road_priority <= 3:
                        break
                    else:
                        print("Please enter a number between 0 and 3")
                except ValueError:
                    print("Please enter a valid number")
            
            day, time_slot = get_current_time_info()
            
            # Use average flows or get from user
            print("\nEnter current flows (8 values) or press Enter for estimated:")
            flow_input = input("Flows (comma-separated): ")
            if flow_input:
                try:
                    flows = [int(x.strip()) for x in flow_input.split(',')]
                    if len(flows) != 8:
                        print("Warning: Expected 8 values, using estimated values instead")
                        raise ValueError
                except ValueError:
                    # Estimate based on time of day
                    hour = datetime.now().hour
                    if 7 <= hour <= 9 or 16 <= hour <= 18:
                        flows = [40, 50, 35, 45, 30, 40, 25, 35]  # Rush hour
                    else:
                        flows = [20, 25, 15, 20, 15, 20, 10, 15]  # Off-peak
            else:
                # Estimate based on time of day
                hour = datetime.now().hour
                if 7 <= hour <= 9 or 16 <= hour <= 18:
                    flows = [40, 50, 35, 45, 30, 40, 25, 35]  # Rush hour
                else:
                    flows = [20, 25, 15, 20, 15, 20, 10, 15]  # Off-peak
            
            green_times = traffic_model.predict_green_times(road_priority, day, time_slot, flows)
            
            print("\nPredicted Green Times:")
            directions = ["North", "East", "South", "West"]
            total_time = sum(green_times)
            for i in range(4):
                print(f"{directions[i]}: {green_times[i]} seconds ({green_times[i]/total_time*100:.1f}%)")
            print(f"Total cycle time: {total_time} seconds")
            
        elif choice == '4':
            print(f"\nModel Statistics:")
            print(f"States in Q-table: {len(traffic_model.Q)}")
            print(f"Training episodes: {len(traffic_model.training_history)}")
            print(f"Action space size: {NUM_ACTIONS}")
            
            if traffic_model.training_history:
                recent_rewards = [h['reward'] for h in traffic_model.training_history[-50:]]
                if recent_rewards:
                    print(f"Average recent reward: {np.mean(recent_rewards):.2f}")
                
        elif choice == '5':
            traffic_model.save_model()
            
        elif choice == '6':
            while True:
                try:
                    hours = int(input("How many hours to simulate? "))
                    if hours > 0:
                        break
                    else:
                        print("Please enter a positive number")
                except ValueError:
                    print("Please enter a valid number")
            auto_collect_demo(traffic_model, duration_hours=hours)
            
        elif choice == '7':
            traffic_model.save_model()
            print("Model saved. Exiting...")
            break
        
        # Auto-save every 5 interactions
        if hasattr(main, 'interaction_count'):
            main.interaction_count += 1
            if main.interaction_count % 5 == 0:
                traffic_model.save_model()
        else:
            main.interaction_count = 1
        
        # Wait 1 second to simulate real-time interval
        time.sleep(1)

if __name__ == "__main__":
    main()