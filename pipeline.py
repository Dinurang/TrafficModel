import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class TrafficSignalOptimizer:
    """
    Complete Deep Learning Pipeline for Traffic Signal Optimization
    Predicts optimal green light times for 4 roads at a junction
    """
    
    def __init__(self, max_cycle_time=120, min_green_time=15, max_green_time=60):
        """
        Initialize the traffic signal optimizer
        
        Args:
            max_cycle_time: Maximum total cycle time in seconds
            min_green_time: Minimum green time per phase in seconds
            max_green_time: Maximum green time per phase in seconds
        """
        self.max_cycle_time = max_cycle_time
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        
        # Initialize components
        self.scaler = StandardScaler()
        self.road_encoder = LabelEncoder()
        self.day_encoder = LabelEncoder()
        self.model = None
        
        # Model configuration
        self.batch_size = 32
        self.epochs = 200
        self.learning_rate = 0.001
        self.patience = 20
        
    def generate_synthetic_data(self, n_samples=50000):
        """
        Generate synthetic training data for traffic signal optimization
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic traffic data
        """
        print(f"Generating {n_samples} synthetic samples...")
        
        data = []
        roads = ['North', 'South', 'East', 'West']
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                'Friday', 'Saturday', 'Sunday']
        priorities = {'Main': 3, 'Secondary': 2, 'Local': 1}
        
        for i in range(n_samples):
            # Generate random timestamp
            base_date = datetime(2024, 1, 1)
            random_days = np.random.randint(0, 365)
            random_hours = np.random.randint(0, 24)
            random_minutes = np.random.randint(0, 60)
            
            timestamp = base_date + timedelta(days=random_days, 
                                             hours=random_hours, 
                                             minutes=random_minutes)
            
            hour = timestamp.hour
            minute = timestamp.minute
            day = days[timestamp.weekday()]
            date_str = timestamp.strftime('%Y-%m-%d')
            
            # Determine if it's peak hour
            is_peak_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
            
            # Generate traffic data for each road
            road_data = {}
            optimal_times = []
            total_flow = 0
            
            for road in roads:
                # Generate realistic traffic patterns
                if is_peak_hour:
                    base_flow = np.random.uniform(30, 80)  # Higher during peak
                else:
                    base_flow = np.random.uniform(10, 40)  # Lower during off-peak
                
                # Add time-of-day variation
                time_factor = 1 + 0.5 * np.sin(2 * np.pi * hour / 24)
                flow_rate = base_flow * time_factor + np.random.normal(0, 5)
                flow_rate = max(flow_rate, 0)
                
                # Determine if vehicles are waiting (higher probability during peak)
                waiting_prob = 0.7 if is_peak_hour else 0.3
                is_waiting = 1 if np.random.random() < waiting_prob else 0
                
                # Assign priority (Main roads get more traffic)
                if road in ['North', 'South']:
                    priority = 'Main'
                else:
                    priority = np.random.choice(['Main', 'Secondary', 'Local'], 
                                              p=[0.3, 0.5, 0.2])
                
                road_data[f'{road}_is_waiting'] = is_waiting
                road_data[f'{road}_flow_rate'] = flow_rate
                road_data[f'{road}_priority'] = priorities[priority]
                road_data[f'{road}_name'] = road
                
                total_flow += flow_rate
            
            # Calculate optimal green times based on traffic conditions
            for road in roads:
                flow_ratio = road_data[f'{road}_flow_rate'] / max(total_flow, 1)
                priority_factor = road_data[f'{road}_priority'] / 3
                waiting_factor = road_data[f'{road}_is_waiting'] * 0.2
                
                # Base allocation
                base_time = self.min_green_time
                
                # Dynamic adjustment
                dynamic_time = (flow_ratio * 0.5 + 
                              priority_factor * 0.3 + 
                              waiting_factor) * (self.max_cycle_time - 4 * self.min_green_time)
                
                optimal_time = base_time + dynamic_time
                
                # Add some randomness
                optimal_time += np.random.normal(0, 2)
                optimal_time = np.clip(optimal_time, self.min_green_time, self.max_green_time)
                
                optimal_times.append(optimal_time)
            
            # Normalize to fit cycle time
            optimal_times = np.array(optimal_times)
            if optimal_times.sum() > self.max_cycle_time:
                optimal_times = optimal_times * (self.max_cycle_time / optimal_times.sum())
            
            # Round to practical values
            optimal_times = np.round(optimal_times / 5) * 5
            
            # Create sample
            sample = {
                'timestamp': timestamp,
                'date': date_str,
                'day': day,
                'time': f"{hour:02d}:{minute:02d}",
                'hour': hour,
                'minute': minute,
                'is_peak_hour': is_peak_hour,
                **road_data,
                'optimal_time_North': optimal_times[0],
                'optimal_time_South': optimal_times[1],
                'optimal_time_East': optimal_times[2],
                'optimal_time_West': optimal_times[3]
            }
            
            data.append(sample)
            
            if (i + 1) % 10000 == 0:
                print(f"Generated {i + 1} samples...")
        
        df = pd.DataFrame(data)
        print(f"Data generation complete. Shape: {df.shape}")
        return df
    
    def prepare_features(self, df):
        """
        Prepare features from raw data
        
        Args:
            df: DataFrame with raw traffic data
            
        Returns:
            X: Feature matrix
            y: Target matrix
            feature_names: List of feature names
        """
        print("Preparing features...")
        
        # Encode categorical features
        df['road_name_encoded'] = self.road_encoder.fit_transform(df['North_name'])
        df['day_encoded'] = self.day_encoder.fit_transform(df['day'])
        
        # Create cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
        df['day_sin'] = np.sin(2 * np.pi * df['day_encoded'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_encoded'] / 7)
        
        # Feature engineering
        roads = ['North', 'South', 'East', 'West']
        
        # 1. Basic features for each road
        basic_features = []
        for road in roads:
            basic_features.extend([
                f'{road}_is_waiting',
                f'{road}_flow_rate',
                f'{road}_priority'
            ])
        
        # 2. Derived features
        derived_features = [
            'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
            'day_sin', 'day_cos', 'is_peak_hour'
        ]
        
        # 3. Interaction features
        df['total_flow'] = sum(df[f'{road}_flow_rate'] for road in roads)
        df['avg_flow'] = df['total_flow'] / 4
        df['max_flow'] = max(df[f'{road}_flow_rate'] for road in roads)
        df['flow_imbalance'] = df['max_flow'] / df['avg_flow']
        
        # 4. Time-based features
        df['morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        df['night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        
        interaction_features = [
            'total_flow', 'avg_flow', 'max_flow', 'flow_imbalance',
            'morning_rush', 'evening_rush', 'night'
        ]
        
        # Combine all features
        feature_columns = basic_features + derived_features + interaction_features
        X = df[feature_columns].values
        
        # Target variables
        target_columns = [f'optimal_time_{road}' for road in roads]
        y = df[target_columns].values
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target matrix shape: {y.shape}")
        
        return X, y, feature_columns
    
    def build_model(self, input_dim):
        """
        Build the deep learning model
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras model
        """
        print("Building deep learning model...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(input_dim,)),
            
            # Hidden layers with batch normalization and dropout
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            
            # Output layer (4 green times)
            layers.Dense(4, activation='relu', name='output')
        ])
        
        # Custom loss function with constraints
        def constrained_loss(y_true, y_pred):
            # Base MSE loss
            mse_loss = keras.losses.mean_squared_error(y_true, y_pred)
            
            # Minimum green time constraint
            min_violation = tf.reduce_mean(
                tf.maximum(self.min_green_time - y_pred, 0)
            )
            
            # Maximum green time constraint
            max_violation = tf.reduce_mean(
                tf.maximum(y_pred - self.max_green_time, 0)
            )
            
            # Total cycle time constraint
            total_time = tf.reduce_sum(y_pred, axis=1)
            cycle_violation = tf.reduce_mean(
                tf.maximum(total_time - self.max_cycle_time, 0)
            )
            
            # Sum constraints violation
            total_violation = min_violation + max_violation + cycle_violation
            
            # Weighted loss
            return mse_loss + 0.1 * total_violation
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=constrained_loss,
            metrics=['mae', 'mse']
        )
        
        model.summary()
        return model
    
    def train(self, X, y, validation_split=0.2):
        """
        Train the model
        
        Args:
            X: Feature matrix
            y: Target matrix
            validation_split: Fraction of data for validation
        """
        print("Training model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Build model
        self.model = self.build_model(X_train.shape[1])
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # Train model
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.training_history = history
        print("Training complete!")
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating model...")
        
        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Apply post-processing constraints
        y_pred_constrained = np.array([self.apply_constraints(pred) for pred in y_pred])
        
        # Calculate metrics
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred_constrained),
            'MSE': mean_squared_error(y_test, y_pred_constrained),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_constrained))
        }
        
        # Calculate constraint violations
        total_times = np.sum(y_pred_constrained, axis=1)
        min_violations = np.sum(y_pred_constrained < self.min_green_time)
        max_violations = np.sum(y_pred_constrained > self.max_green_time)
        cycle_violations = np.sum(total_times > self.max_cycle_time)
        
        metrics['min_violations_pct'] = (min_violations / (y_pred_constrained.size)) * 100
        metrics['max_violations_pct'] = (max_violations / (y_pred_constrained.size)) * 100
        metrics['cycle_violations_pct'] = (cycle_violations / len(y_pred_constrained)) * 100
        
        print("\nEvaluation Metrics:")
        print(f"MAE: {metrics['MAE']:.2f} seconds")
        print(f"RMSE: {metrics['RMSE']:.2f} seconds")
        print(f"Minimum time violations: {metrics['min_violations_pct']:.2f}%")
        print(f"Maximum time violations: {metrics['max_violations_pct']:.2f}%")
        print(f"Cycle time violations: {metrics['cycle_violations_pct']:.2f}%")
        
        return metrics, y_pred_constrained
    
    def apply_constraints(self, times):
        """
        Apply practical constraints to predicted times
        
        Args:
            times: Predicted green times
            
        Returns:
            Constrained times
        """
        times = np.array(times)
        
        # Ensure minimum green time
        times = np.maximum(times, self.min_green_time)
        
        # Ensure maximum green time
        times = np.minimum(times, self.max_green_time)
        
        # Adjust total cycle time
        total = np.sum(times)
        if total > self.max_cycle_time:
            times = times * (self.max_cycle_time / total)
        
        # Round to practical values (multiples of 5 seconds)
        times = np.round(times / 5) * 5
        
        # Final bounds check
        times = np.clip(times, self.min_green_time, self.max_green_time)
        
        return times
    
    def predict_single(self, traffic_data):
        """
        Predict optimal times for a single traffic scenario
        
        Args:
            traffic_data: Dictionary with current traffic data
            
        Returns:
            Optimized green times for each road
        """
        # Convert to DataFrame for feature preparation
        df = pd.DataFrame([traffic_data])
        
        # Prepare features (single sample)
        X, _, _ = self.prepare_features(df)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        raw_prediction = self.model.predict(X_scaled, verbose=0)[0]
        
        # Apply constraints
        optimized_times = self.apply_constraints(raw_prediction)
        
        return optimized_times
    
    def plot_training_history(self):
        """Plot training history"""
        if not hasattr(self, 'training_history'):
            print("No training history available")
            return
        
        history = self.training_history.history
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(history['loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot MAE
        axes[1].plot(history['mae'], label='Training MAE')
        axes[1].plot(history['val_mae'], label='Validation MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE (seconds)')
        axes[1].set_title('Training and Validation MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions_vs_actual(self, y_true, y_pred):
        """Plot predictions vs actual values"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        roads = ['North', 'South', 'East', 'West']
        
        for idx, (ax, road) in enumerate(zip(axes.flat, roads)):
            ax.scatter(y_true[:, idx], y_pred[:, idx], alpha=0.5)
            ax.plot([self.min_green_time, self.max_green_time], 
                   [self.min_green_time, self.max_green_time], 
                   'r--', label='Perfect Prediction')
            ax.set_xlabel(f'Actual {road} Time (s)')
            ax.set_ylabel(f'Predicted {road} Time (s)')
            ax.set_title(f'{road} Road: Predictions vs Actual')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, path='traffic_signal_model'):
        """Save the model and preprocessing objects"""
        import joblib
        
        # Save model
        self.model.save(f'{path}.h5')
        
        # Save preprocessing objects
        joblib.dump({
            'scaler': self.scaler,
            'road_encoder': self.road_encoder,
            'day_encoder': self.day_encoder
        }, f'{path}_preprocessors.pkl')
        
        print(f"Model saved to {path}.h5")
        print(f"Preprocessors saved to {path}_preprocessors.pkl")
    
    def load_model(self, path='traffic_signal_model'):
        """Load the model and preprocessing objects"""
        import joblib
        
        # Load model
        self.model = keras.models.load_model(f'{path}.h5', 
                                           custom_objects={'constrained_loss': None})
        
        # Load preprocessing objects
        preprocessors = joblib.load(f'{path}_preprocessors.pkl')
        self.scaler = preprocessors['scaler']
        self.road_encoder = preprocessors['road_encoder']
        self.day_encoder = preprocessors['day_encoder']
        
        print("Model loaded successfully!")


# Example usage and testing
def main():
    # Initialize the optimizer
    optimizer = TrafficSignalOptimizer(
        max_cycle_time=120,
        min_green_time=15,
        max_green_time=60
    )
    
    # Step 1: Generate synthetic data
    print("=" * 50)
    print("STEP 1: Data Generation")
    print("=" * 50)
    data = optimizer.generate_synthetic_data(n_samples=20000)
    
    # Step 2: Prepare features
    print("\n" + "=" * 50)
    print("STEP 2: Feature Preparation")
    print("=" * 50)
    X, y, feature_names = optimizer.prepare_features(data)
    
    # Step 3: Train model
    print("\n" + "=" * 50)
    print("STEP 3: Model Training")
    print("=" * 50)
    history = optimizer.train(X, y, validation_split=0.2)
    
    # Step 4: Evaluate model
    print("\n" + "=" * 50)
    print("STEP 4: Model Evaluation")
    print("=" * 50)
    
    # Split test data (using the same data for demonstration)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    metrics, y_pred = optimizer.evaluate(X_test, y_test)
    
    # Step 5: Visualizations
    print("\n" + "=" * 50)
    print("STEP 5: Visualizations")
    print("=" * 50)
    optimizer.plot_training_history()
    optimizer.plot_predictions_vs_actual(y_test[:100], y_pred[:100])
    
    # Step 6: Real-time prediction example
    print("\n" + "=" * 50)
    print("STEP 6: Real-time Prediction Example")
    print("=" * 50)
    
    # Create a sample traffic scenario
    sample_traffic = {
        'timestamp': datetime(2024, 1, 15, 8, 30),  # Monday morning peak
        'date': '2024-01-15',
        'day': 'Monday',
        'time': '08:30',
        'hour': 8,
        'minute': 30,
        'is_peak_hour': 1,
        'North_is_waiting': 1,
        'North_flow_rate': 65.2,
        'North_priority': 3,
        'North_name': 'North',
        'South_is_waiting': 1,
        'South_flow_rate': 58.7,
        'South_priority': 3,
        'South_name': 'South',
        'East_is_waiting': 0,
        'East_flow_rate': 22.3,
        'East_priority': 2,
        'East_name': 'East',
        'West_is_waiting': 1,
        'West_flow_rate': 31.5,
        'West_priority': 2,
        'West_name': 'West'
    }
    
    # Get optimized green times
    optimized_times = optimizer.predict_single(sample_traffic)
    
    print("\nSample Traffic Scenario:")
    print(f"Time: {sample_traffic['time']} on {sample_traffic['day']}")
    print("Traffic Conditions:")
    print(f"  North: Waiting={sample_traffic['North_is_waiting']}, "
          f"Flow={sample_traffic['North_flow_rate']:.1f} vehicles/min, "
          f"Priority={sample_traffic['North_priority']}")
    print(f"  South: Waiting={sample_traffic['South_is_waiting']}, "
          f"Flow={sample_traffic['South_flow_rate']:.1f} vehicles/min, "
          f"Priority={sample_traffic['South_priority']}")
    print(f"  East:  Waiting={sample_traffic['East_is_waiting']}, "
          f"Flow={sample_traffic['East_flow_rate']:.1f} vehicles/min, "
          f"Priority={sample_traffic['East_priority']}")
    print(f"  West:  Waiting={sample_traffic['West_is_waiting']}, "
          f"Flow={sample_traffic['West_flow_rate']:.1f} vehicles/min, "
          f"Priority={sample_traffic['West_priority']}")
    
    print("\nOptimized Green Times:")
    roads = ['North', 'South', 'East', 'West']
    for road, time in zip(roads, optimized_times):
        print(f"  {road} road: {time:.0f} seconds")
    
    print(f"\nTotal cycle time: {np.sum(optimized_times):.0f} seconds")
    
    # Step 7: Save the model
    print("\n" + "=" * 50)
    print("STEP 7: Save Model")
    print("=" * 50)
    optimizer.save_model('traffic_signal_optimizer')
    
    return optimizer, metrics


if __name__ == "__main__":
    # Run the complete pipeline
    optimizer, metrics = main()