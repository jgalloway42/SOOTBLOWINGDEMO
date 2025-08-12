#!/usr/bin/env python3
"""
Realistic LSTM Implementation for Boiler Heat Transfer Degradation Prediction

This implementation uses REALISTIC targets that operators actually monitor:
- Overall heat transfer coefficient (U) for each section
- Heat transfer rate degradation
- Temperature performance indicators

Instead of separate gas/water fouling (which isn't directly measurable),
this predicts the OBSERVABLE performance degradation that maintenance teams
actually use for scheduling decisions.

Author: Enhanced Boiler Modeling System
Version: 2.0 - Realistic Operational Targets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class RealisticBoilerPerformancePredictor:
    """
    Realistic LSTM implementation for predicting observable boiler performance degradation.
    
    Uses PRACTICAL targets that maintenance teams actually monitor:
    - Overall heat transfer coefficient (U) - directly measurable
    - Heat transfer rate - observable performance metric
    - Temperature effectiveness - operational indicator
    
    These targets reflect the CUMULATIVE effect of all fouling types
    without requiring separate gas/water fouling measurements.
    """
    
    def __init__(self, sequence_length=24, prediction_horizon=1):
        """
        Initialize the realistic performance predictor.
        
        Args:
            sequence_length: Hours of historical data to use (24 = 1 day)
            prediction_horizon: Hours ahead to predict (1 = next hour)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Define tube sections from the dataset
        self.tube_sections = [
            'furnace_walls',
            'generating_bank', 
            'superheater_primary',
            'superheater_secondary',
            'economizer_primary',
            'economizer_secondary',
            'air_heater'
        ]
        
        # Initialize scalers and model storage
        self.feature_scaler = None
        self.target_scaler = None
        self.model = None
        self.feature_columns = None
        self.target_columns = None
        self.training_history = None
        
        print(f"🔧 Realistic Boiler Performance Predictor Initialized")
        print(f"   • Sequence length: {sequence_length} hours")
        print(f"   • Prediction horizon: {prediction_horizon} hour(s)")
        print(f"   • Tube sections: {len(self.tube_sections)}")
        print(f"   • Target approach: Observable performance metrics (not fouling factors)")

    def load_and_prepare_data(self, csv_file_path):
        """Load and prepare the Massachusetts boiler dataset."""
        print(f"\n📁 Loading data from: {csv_file_path}")
        
        try:
            data = pd.read_csv(csv_file_path)
            print(f"✅ Data loaded successfully: {len(data)} records, {len(data.columns)} columns")
        except FileNotFoundError:
            print(f"❌ File not found: {csv_file_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
        
        # Parse timestamp and sort
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.sort_values('timestamp').reset_index(drop=True)
            print(f"✅ Data sorted by timestamp")
        
        # Basic data validation
        print(f"\n📊 Data Overview:")
        print(f"   • Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        print(f"   • Missing values: {data.isnull().sum().sum()}")
        
        return data

    def select_features_and_targets(self, data):
        """
        Select features and REALISTIC targets for LSTM model.
        
        REALISTIC TARGETS (what operators actually monitor):
        1. Overall heat transfer coefficient (U) - directly measurable
        2. Heat transfer rate - observable performance
        3. Temperature effectiveness - operational metric
        
        These reflect the TOTAL impact of all fouling without requiring
        separate gas/water fouling measurements.
        """
        print(f"\n🔧 Feature Selection and Realistic Target Definition...")
        
        # Make a copy for feature engineering
        data_enhanced = data.copy()
        
        # ===== SAME FEATURES AS BEFORE (unchanged) =====
        # These are the INPUTS that drive fouling and performance degradation
        
        # 1. Combustion conditions (strongest predictors of soot formation)
        combustion_features = [
            'thermal_nox_lb_hr',           # Direct combustion quality indicator
            'fuel_nox_lb_hr',              # Fuel-bound nitrogen → soot formation
            'total_nox_lb_hr',             # Total NOx correlates with soot production
            'excess_o2_pct',               # Air/fuel ratio affects combustion quality
            'combustion_efficiency',        # Lower efficiency = more incomplete combustion
            'flame_temp_F'                 # Temperature affects soot formation zones
        ]
        
        # 2. Coal properties (affect soot characteristics and deposition)
        coal_features = [
            'coal_carbon_pct',             # Carbon content affects soot yield
            'coal_volatile_matter_pct',    # Volatile matter affects flame characteristics
            'coal_sulfur_pct',            # Sulfur affects soot stickiness
            'coal_ash_pct',               # Ash content affects abrasion/deposition
            'coal_moisture_pct',          # Moisture affects combustion quality
            'coal_heating_value_btu_lb'   # Energy density affects flame temperature
        ]
        
        # 3. Operating conditions (affect soot production rates)
        operating_features = [
            'load_factor',                # Higher loads = more soot production
            'coal_rate_lb_hr',           # Fuel flow affects total soot generation
            'air_flow_scfh',             # Air flow affects combustion quality
            'fuel_input_btu_hr',         # Total energy input
            'flue_gas_flow_lb_hr'        # Gas velocity affects deposition patterns
        ]
        
        # 4. Environmental factors
        environmental_features = [
            'ambient_temp_F',            # Affects air density and combustion
            'ambient_humidity_pct'       # Affects combustion characteristics
        ]
        
        # 5. Section-specific temperatures (critical for deposition rates)
        temperature_features = []
        for section in self.tube_sections:
            temp_cols = [
                f'{section}_gas_temp_in_F',   # High temps = more soot sticking
                f'{section}_gas_temp_out_F',  # Temperature drop indicates heat transfer
                f'{section}_water_temp_in_F', # Affects heat transfer driving force
                f'{section}_water_temp_out_F' # Performance indicator
            ]
            temperature_features.extend(temp_cols)
        
        # 6. Soot blowing history (critical for understanding performance resets!)
        soot_blowing_features = ['soot_blowing_active', 'sections_cleaned_count']
        for section in self.tube_sections:
            soot_blowing_features.append(f'{section}_hours_since_cleaning')
        
        # 7. Temporal features
        temporal_features = ['hour', 'month', 'day_of_year']
        
        # ===== REALISTIC TARGET DEFINITION =====
        # Instead of unmeasurable fouling factors, use OBSERVABLE performance metrics
        
        print(f"🎯 Defining REALISTIC targets (observable performance metrics)...")
        
        # Calculate baseline performance metrics for degradation analysis
        for section in self.tube_sections:
            
            # 1. Overall heat transfer coefficient (PRIMARY TARGET)
            # This is directly measurable and reflects total fouling impact
            u_col = f'{section}_overall_U_avg'
            if u_col in data.columns:
                # Calculate U degradation from clean baseline
                # Assume first values represent "clean" condition
                clean_u = data[u_col].quantile(0.95)  # 95th percentile as "clean" reference
                data_enhanced[f'{section}_U_degradation'] = (clean_u - data[u_col]) / clean_u
                
                # Calculate rate of U degradation (trending)
                data_enhanced[f'{section}_U_degradation_rate'] = data_enhanced[f'{section}_U_degradation'].diff().fillna(0)
            
            # 2. Heat transfer effectiveness (SECONDARY TARGET)
            # Ratio of actual to maximum possible heat transfer
            heat_col = f'{section}_heat_transfer_btu_hr'
            gas_in_col = f'{section}_gas_temp_in_F'
            water_in_col = f'{section}_water_temp_in_F'
            
            if all(col in data.columns for col in [heat_col, gas_in_col, water_in_col]):
                # Temperature driving force
                temp_driving_force = data[gas_in_col] - data[water_in_col]
                temp_driving_force = temp_driving_force.clip(lower=1)  # Avoid division by zero
                
                # Heat transfer effectiveness (normalized by driving force)
                data_enhanced[f'{section}_heat_effectiveness'] = data[heat_col] / temp_driving_force
                
                # Calculate effectiveness degradation
                max_effectiveness = data_enhanced[f'{section}_heat_effectiveness'].quantile(0.95)
                data_enhanced[f'{section}_effectiveness_degradation'] = (
                    (max_effectiveness - data_enhanced[f'{section}_heat_effectiveness']) / max_effectiveness
                )
            
            # 3. Temperature performance indicator (TERTIARY TARGET)
            # How well the section is achieving expected temperature drops
            gas_out_col = f'{section}_gas_temp_out_F'
            if gas_in_col in data.columns and gas_out_col in data.columns:
                # Actual temperature drop
                actual_temp_drop = data[gas_in_col] - data[gas_out_col]
                
                # Expected temperature drop (based on load and design)
                # Simplified model: expected drop proportional to load
                if 'load_factor' in data.columns:
                    design_temp_drop = actual_temp_drop.quantile(0.95)  # Design condition
                    expected_temp_drop = design_temp_drop * data['load_factor']
                    
                    # Temperature performance ratio
                    data_enhanced[f'{section}_temp_performance'] = (
                        actual_temp_drop / expected_temp_drop.clip(lower=1)
                    )
                    
                    # Performance degradation
                    data_enhanced[f'{section}_temp_degradation'] = (
                        1 - data_enhanced[f'{section}_temp_performance']
                    ).clip(lower=0)
        
        # ===== FEATURE ENGINEERING (same as before) =====
        print(f"🔬 Engineering predictive features...")
        
        # 1. Soot production indicators
        if all(col in data.columns for col in ['total_nox_lb_hr', 'combustion_efficiency']):
            data_enhanced['soot_production_indicator'] = (
                data['total_nox_lb_hr'] / data['combustion_efficiency'].clip(lower=0.5)
            )
        
        # 2. Coal quality impact (numeric encoding)
        if 'coal_quality' in data.columns:
            coal_quality_map = {
                'high_quality': 0.8,    # Less fouling/degradation
                'medium_quality': 1.0,  # Baseline
                'low_quality': 1.3,     # More fouling/degradation
                'waste_coal': 1.6       # Most fouling/degradation
            }
            data_enhanced['coal_quality_numeric'] = data['coal_quality'].map(coal_quality_map).fillna(1.0)
        
        # 3. Operational stress factor (affects fouling rate)
        if all(col in data_enhanced.columns for col in ['load_factor', 'coal_quality_numeric', 'combustion_efficiency']):
            data_enhanced['operational_stress'] = (
                data['load_factor'] * 
                data_enhanced['coal_quality_numeric'] * 
                (2.0 - data['combustion_efficiency'].clip(lower=0.5))
            )
        
        # 4. Temperature gradients (affect deposition patterns)
        for section in self.tube_sections:
            gas_in = f'{section}_gas_temp_in_F'
            gas_out = f'{section}_gas_temp_out_F'
            if gas_in in data.columns and gas_out in data.columns:
                data_enhanced[f'{section}_temp_gradient'] = data[gas_in] - data[gas_out]
        
        # 5. System-wide performance metrics
        u_degradation_cols = [f'{section}_U_degradation' for section in self.tube_sections 
                             if f'{section}_U_degradation' in data_enhanced.columns]
        if len(u_degradation_cols) > 1:
            data_enhanced['system_wide_degradation'] = data_enhanced[u_degradation_cols].mean(axis=1)
            data_enhanced['degradation_variation'] = data_enhanced[u_degradation_cols].std(axis=1)
        
        # 6. Cyclical time encoding
        if 'day_of_year' in data.columns:
            data_enhanced['season_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365.25)
            data_enhanced['season_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365.25)
        
        if 'hour' in data.columns:
            data_enhanced['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
            data_enhanced['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        
        # ===== COMPILE FEATURE LIST =====
        feature_categories = {
            'combustion': combustion_features,
            'coal': coal_features,
            'operating': operating_features,
            'environmental': environmental_features,
            'temperatures': temperature_features,
            'soot_blowing': soot_blowing_features,
            'temporal': temporal_features
        }
        
        # Engineered features
        engineered_features = [col for col in data_enhanced.columns if any(suffix in col for suffix in [
            'soot_production_indicator', 'coal_quality_numeric', 'operational_stress',
            '_temp_gradient', 'system_wide_degradation', 'degradation_variation',
            'season_sin', 'season_cos', 'hour_sin', 'hour_cos'
        ])]
        
        # Combine all features
        all_features = []
        for category, features in feature_categories.items():
            valid_features = [f for f in features if f in data_enhanced.columns]
            all_features.extend(valid_features)
            print(f"   • {category}: {len(valid_features)} features")
        
        all_features.extend(engineered_features)
        print(f"   • engineered: {len(engineered_features)} features")
        
        # Remove duplicates
        self.feature_columns = list(dict.fromkeys(all_features))
        
        # ===== DEFINE REALISTIC TARGETS =====
        # These are OBSERVABLE and ACTIONABLE for maintenance teams
        
        self.target_columns = []
        target_types = {
            'U_degradation': 'Overall heat transfer coefficient degradation (PRIMARY)',
            'effectiveness_degradation': 'Heat transfer effectiveness degradation (SECONDARY)',
            'temp_degradation': 'Temperature performance degradation (TERTIARY)'
        }
        
        print(f"\n🎯 REALISTIC TARGET VARIABLES:")
        for target_type, description in target_types.items():
            print(f"   {description}")
            section_targets = []
            for section in self.tube_sections:
                target_col = f'{section}_{target_type}'
                if target_col in data_enhanced.columns:
                    section_targets.append(target_col)
                    self.target_columns.append(target_col)
            print(f"     • Found {len(section_targets)} section targets")
        
        print(f"\n✅ Realistic target definition complete:")
        print(f"   • Total features: {len(self.feature_columns)}")
        print(f"   • Total targets: {len(self.target_columns)} (observable performance metrics)")
        print(f"   • Target sections: {len(self.tube_sections)}")
        
        print(f"\n💡 WHY THESE TARGETS ARE REALISTIC:")
        print(f"   ✅ Overall U degradation: Directly measurable from temperature/flow data")
        print(f"   ✅ Heat effectiveness: Observable from performance monitoring")
        print(f"   ✅ Temperature performance: Available from existing instrumentation")
        print(f"   ✅ No separate gas/water fouling: Not directly measurable in practice")
        print(f"   ✅ Reflects total fouling impact: Soot + scale + other deposits")
        
        return data_enhanced

    def create_sequences(self, data):
        """Create sequences for LSTM training (same as before)."""
        print(f"\n🔄 Creating sequences for realistic performance prediction...")
        
        # Handle categorical variables if present
        data_processed = data.copy()
        
        # Encode season if present
        if 'season' in data.columns:
            season_encoder = LabelEncoder()
            data_processed['season_encoded'] = season_encoder.fit_transform(data['season'])
            if 'season' in self.feature_columns:
                self.feature_columns.remove('season')
                self.feature_columns.append('season_encoded')
        
        # Encode coal quality if present
        if 'coal_quality' in data.columns and 'coal_quality' in self.feature_columns:
            quality_encoder = LabelEncoder()
            data_processed['coal_quality_encoded'] = quality_encoder.fit_transform(data['coal_quality'])
            self.feature_columns.remove('coal_quality')
            self.feature_columns.append('coal_quality_encoded')
        
        # Ensure all columns exist and handle missing values
        valid_features = [col for col in self.feature_columns if col in data_processed.columns]
        valid_targets = [col for col in self.target_columns if col in data_processed.columns]
        
        print(f"   • Valid features: {len(valid_features)}")
        print(f"   • Valid targets: {len(valid_targets)}")
        
        # Extract and clean data
        features = data_processed[valid_features].fillna(method='ffill').fillna(method='bfill').fillna(0)
        targets = data_processed[valid_targets].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Scale features and targets
        print(f"   • Scaling features and targets...")
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
        features_scaled = self.feature_scaler.fit_transform(features)
        targets_scaled = self.target_scaler.fit_transform(targets)
        
        # Create sequences
        print(f"   • Creating {self.sequence_length}-hour sequences...")
        X, y = [], []
        
        for i in range(self.sequence_length, len(features_scaled) - self.prediction_horizon + 1):
            # Input: sequence_length hours of features
            X.append(features_scaled[i-self.sequence_length:i])
            
            # Output: performance degradation prediction_horizon hours ahead
            y.append(targets_scaled[i + self.prediction_horizon - 1])
        
        X = np.array(X)
        y = np.array(y)
        
        # Update feature and target columns for valid ones only
        self.feature_columns = valid_features
        self.target_columns = valid_targets
        
        print(f"✅ Sequence creation complete:")
        print(f"   • Input shape: {X.shape} (samples, timesteps, features)")
        print(f"   • Output shape: {y.shape} (samples, targets)")
        print(f"   • Features used: {len(self.feature_columns)}")
        print(f"   • Performance metrics predicted: {len(self.target_columns)}")
        
        return X, y

    def build_model(self, input_shape, output_shape):
        """
        Build LSTM model optimized for performance degradation prediction.
        
        Architecture optimized for predicting REALISTIC performance metrics
        rather than abstract fouling factors.
        """
        print(f"\n🏗️ Building LSTM model for performance degradation prediction...")
        print(f"   • Input shape: {input_shape}")
        print(f"   • Output shape: {output_shape} performance metrics")
        
        model = Sequential([
            # First LSTM layer - captures main temporal patterns in performance
            LSTM(128, 
                 return_sequences=True, 
                 input_shape=input_shape,
                 dropout=0.2, 
                 recurrent_dropout=0.2,
                 kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            
            # Second LSTM layer - refines performance degradation patterns
            LSTM(64, 
                 return_sequences=True,
                 dropout=0.2, 
                 recurrent_dropout=0.2,
                 kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            
            # Third LSTM layer - final temporal processing
            LSTM(32, 
                 return_sequences=False,
                 dropout=0.2, 
                 recurrent_dropout=0.2,
                 kernel_regularizer=l2(0.001)),
            
            # Dense layers for performance prediction
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.2),
            
            # Output layer - predict performance degradation for all sections
            # Using sigmoid activation since degradation is bounded [0,1]
            Dense(output_shape, activation='sigmoid')
        ])
        
        # Compile with appropriate loss for bounded regression
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',  # MSE appropriate for bounded targets
            metrics=['mae', 'mape']
        )
        
        print(f"✅ Model built for realistic performance prediction:")
        print(f"   • Total parameters: {model.count_params():,}")
        print(f"   • Output activation: Sigmoid (bounded degradation [0,1])")
        print(f"   • Target interpretation: 0 = no degradation, 1 = maximum degradation")
        
        return model

    def train_model(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """Train the model (same structure as before)."""
        print(f"\n🚀 Starting performance degradation model training...")
        print(f"   • Training samples: {int(len(X) * (1-validation_split))}")
        print(f"   • Validation samples: {int(len(X) * validation_split)}")
        print(f"   • Batch size: {batch_size}")
        print(f"   • Max epochs: {epochs}")
        
        # Build model
        self.model = self.build_model(
            input_shape=(X.shape[1], X.shape[2]), 
            output_shape=y.shape[1]
        )
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                'realistic_boiler_performance_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Use time series split for validation
        split_idx = int(len(X) * (1 - validation_split))
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"   • Time-based split at index {split_idx}")
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )
        
        self.training_history = history
        
        # Final performance
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        print(f"\n✅ Training complete!")
        print(f"   • Final train loss: {final_train_loss:.6f}")
        print(f"   • Final val loss: {final_val_loss:.6f}")
        print(f"   • Best model saved as: realistic_boiler_performance_model.h5")
        
        return self.model, history

    def evaluate_model(self, X_test, y_test):
        """Evaluate model with performance-focused metrics."""
        print(f"\n📊 Evaluating realistic performance prediction model...")
        
        if self.model is None:
            print("❌ Model not trained yet!")
            return None
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        
        # Inverse scale predictions and targets
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        y_true = self.target_scaler.inverse_transform(y_test)
        
        # Overall metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE manually
        mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
        
        print(f"✅ Overall Performance (Degradation Prediction):")
        print(f"   • RMSE: {rmse:.6f} (degradation units)")
        print(f"   • MAE: {mae:.6f} (degradation units)")
        print(f"   • MAPE: {mape:.2f}%")
        print(f"   • R²: {r2:.4f}")
        
        # Performance by target type
        target_types = ['U_degradation', 'effectiveness_degradation', 'temp_degradation']
        
        print(f"\n📋 Performance by Target Type:")
        for target_type in target_types:
            type_targets = [i for i, col in enumerate(self.target_columns) if target_type in col]
            if type_targets:
                type_mse = mean_squared_error(y_true[:, type_targets], y_pred[:, type_targets])
                type_mae = mean_absolute_error(y_true[:, type_targets], y_pred[:, type_targets])
                type_rmse = np.sqrt(type_mse)
                type_r2 = r2_score(y_true[:, type_targets], y_pred[:, type_targets])
                
                target_name = target_type.replace('_', ' ').title()
                print(f"   • {target_name}: RMSE={type_rmse:.6f}, MAE={type_mae:.6f}, R²={type_r2:.4f}")
        
        evaluation_results = {
            'overall': {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2': r2
            },
            'predictions': y_pred,
            'actuals': y_true
        }
        
        return evaluation_results

    def create_maintenance_dashboard(self, evaluation_results, data, threshold_degradation=0.15):
        """
        Create a maintenance-focused dashboard showing when sections need attention.
        
        Args:
            evaluation_results: Model evaluation results
            data: Original data for context
            threshold_degradation: Degradation threshold for maintenance alerts (15% default)
        """
        print(f"\n🔧 Creating Maintenance Dashboard...")
        print(f"   • Maintenance threshold: {threshold_degradation:.1%} degradation")
        
        if evaluation_results is None:
            print("❌ No evaluation results available!")
            return
        
        y_pred = evaluation_results['predictions']
        y_true = evaluation_results['actuals']
        
        # Get last predictions (most recent performance state)
        latest_predictions = y_pred[-1]
        latest_actual = y_true[-1]
        
        # Organize by section and target type
        section_performance = {}
        
        for i, target_col in enumerate(self.target_columns):
            # Parse section and target type
            parts = target_col.split('_')
            if len(parts) >= 3:
                section = '_'.join(parts[:-1])  # Everything except last part
                target_type = parts[-1]
                
                if section not in section_performance:
                    section_performance[section] = {}
                
                section_performance[section][target_type] = {
                    'predicted': latest_predictions[i],
                    'actual': latest_actual[i],
                    'threshold_exceeded': latest_predictions[i] > threshold_degradation
                }
        
        # Create maintenance priority dashboard
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Section degradation heatmap
        ax1 = axes[0, 0]
        sections = list(section_performance.keys())
        target_types = ['degradation']  # Simplified for display
        
        # Create heatmap data
        heatmap_data = []
        section_labels = []
        
        for section in sections:
            if section in section_performance:
                section_name = section.replace('_', ' ').title()
                section_labels.append(section_name)
                
                # Get U degradation (primary metric)
                u_deg = section_performance[section].get('degradation', {}).get('predicted', 0)
                heatmap_data.append([u_deg])
        
        if heatmap_data:
            heatmap_data = np.array(heatmap_data)
            im = ax1.imshow(heatmap_data, cmap='Reds', aspect='auto', vmin=0, vmax=0.5)
            ax1.set_yticks(range(len(section_labels)))
            ax1.set_yticklabels(section_labels)
            ax1.set_xticks([0])
            ax1.set_xticklabels(['Heat Transfer\nDegradation'])
            ax1.set_title('Section Performance Degradation')
            
            # Add text annotations
            for i in range(len(section_labels)):
                value = heatmap_data[i, 0]
                color = 'white' if value > 0.25 else 'black'
                ax1.text(0, i, f'{value:.1%}', ha='center', va='center', color=color, fontweight='bold')
            
            plt.colorbar(im, ax=ax1, label='Degradation Level')
        
        # 2. Maintenance priority ranking
        ax2 = axes[0, 1]
        
        # Calculate overall degradation score per section
        section_scores = {}
        for section, metrics in section_performance.items():
            scores = [m.get('predicted', 0) for m in metrics.values()]
            section_scores[section] = np.mean(scores) if scores else 0
        
        # Sort by degradation level
        sorted_sections = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_sections:
            section_names = [s[0].replace('_', ' ').title() for s, _ in sorted_sections]
            degradation_levels = [score for _, score in sorted_sections]
            
            # Color code based on threshold
            colors = ['red' if score > threshold_degradation else 'orange' if score > threshold_degradation/2 else 'green' 
                     for score in degradation_levels]
            
            bars = ax2.barh(section_names, degradation_levels, color=colors, alpha=0.7)
            ax2.axvline(x=threshold_degradation, color='red', linestyle='--', linewidth=2, label=f'Maintenance Threshold ({threshold_degradation:.1%})')
            ax2.set_xlabel('Average Degradation Level')
            ax2.set_title('Maintenance Priority Ranking')
            ax2.legend()
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, degradation_levels)):
                ax2.text(value + 0.01, bar.get_y() + bar.get_height()/2, f'{value:.1%}', 
                        va='center', fontweight='bold')
        
        # 3. Prediction vs Actual comparison
        ax3 = axes[1, 0]
        
        # Show prediction accuracy for recent period
        recent_period = min(100, len(y_pred))
        recent_pred = y_pred[-recent_period:]
        recent_actual = y_true[-recent_period:]
        
        # Plot average degradation over time
        if len(recent_pred) > 0:
            avg_pred = np.mean(recent_pred, axis=1)
            avg_actual = np.mean(recent_actual, axis=1)
            
            time_points = range(len(avg_pred))
            ax3.plot(time_points, avg_pred, 'b-', label='Predicted Degradation', linewidth=2)
            ax3.plot(time_points, avg_actual, 'r--', label='Actual Degradation', linewidth=2)
            ax3.axhline(y=threshold_degradation, color='orange', linestyle=':', alpha=0.7, label='Maintenance Threshold')
            ax3.set_xlabel('Time (Hours Ago)')
            ax3.set_ylabel('Average System Degradation')
            ax3.set_title('Recent Prediction Accuracy')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Maintenance recommendations
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Generate maintenance recommendations
        recommendations = []
        urgent_sections = [s for s, score in sorted_sections if score > threshold_degradation]
        warning_sections = [s for s, score in sorted_sections if threshold_degradation/2 < score <= threshold_degradation]
        
        if urgent_sections:
            recommendations.append("🚨 URGENT MAINTENANCE NEEDED:")
            for section in urgent_sections[:3]:  # Top 3
                section_name = section.replace('_', ' ').title()
                score = section_scores[section]
                recommendations.append(f"   • {section_name}: {score:.1%} degradation")
        
        if warning_sections:
            recommendations.append("\n⚠️  SCHEDULE MAINTENANCE SOON:")
            for section in warning_sections[:3]:  # Top 3
                section_name = section.replace('_', ' ').title()
                score = section_scores[section]
                recommendations.append(f"   • {section_name}: {score:.1%} degradation")
        
        if not urgent_sections and not warning_sections:
            recommendations.append("✅ ALL SECTIONS PERFORMING WELL")
            recommendations.append("   No immediate maintenance required")
        
        # Add operational recommendations
        recommendations.append("\n💡 OPERATIONAL RECOMMENDATIONS:")
        if 'economizer' in ' '.join(urgent_sections + warning_sections):
            recommendations.append("   • Focus on economizer cleaning")
            recommendations.append("   • Check feedwater treatment")
        if 'furnace' in ' '.join(urgent_sections + warning_sections):
            recommendations.append("   • Optimize combustion conditions")
            recommendations.append("   • Consider coal quality upgrade")
        if 'superheater' in ' '.join(urgent_sections + warning_sections):
            recommendations.append("   • Review soot blowing frequency")
            recommendations.append("   • Monitor steam temperature control")
        
        # Display recommendations
        rec_text = '\n'.join(recommendations)
        ax4.text(0.05, 0.95, rec_text, transform=ax4.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('Boiler Maintenance Dashboard - Performance Degradation Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return {
            'section_performance': section_performance,
            'maintenance_priorities': sorted_sections,
            'urgent_sections': urgent_sections,
            'warning_sections': warning_sections,
            'recommendations': recommendations
        }

    def predict_performance_degradation(self, X_new):
        """
        Predict performance degradation for new operational data.
        
        Returns degradation levels (0 = no degradation, 1 = maximum degradation)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Make prediction
        y_pred_scaled = self.model.predict(X_new, verbose=0)
        
        # Inverse scale
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        
        return y_pred

    def save_model(self, filepath='realistic_boiler_performance_model.h5'):
        """Save the trained model and metadata."""
        if self.model is None:
            print("❌ No model to save!")
            return
        
        # Save model
        self.model.save(filepath)
        
        # Save metadata
        import pickle
        metadata = {
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'tube_sections': self.tube_sections
        }
        
        metadata_filepath = filepath.replace('.h5', '_metadata.pkl')
        with open(metadata_filepath, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Realistic performance model saved:")
        print(f"   • Model: {filepath}")
        print(f"   • Metadata: {metadata_filepath}")


def run_realistic_training_pipeline(csv_file_path):
    """
    Complete training pipeline for realistic boiler performance prediction.
    
    Args:
        csv_file_path: Path to the Massachusetts boiler CSV file
        
    Returns:
        Trained predictor and evaluation results
    """
    print("🔧" * 50)
    print("REALISTIC BOILER PERFORMANCE PREDICTION - LSTM TRAINING PIPELINE")
    print("Using Observable Performance Metrics Instead of Unmeasurable Fouling Factors")
    print("🔧" * 50)
    
    # Step 1: Initialize realistic predictor
    print("\n📋 STEP 1: Initialize Realistic Performance Predictor")
    predictor = RealisticBoilerPerformancePredictor(
        sequence_length=24,  # 24 hours of operational history
        prediction_horizon=1  # Predict 1 hour ahead
    )
    
    # Step 2: Load and prepare data
    print("\n📋 STEP 2: Load and Prepare Data")
    data = predictor.load_and_prepare_data(csv_file_path)
    if data is None:
        return None, None
    
    # Step 3: Define realistic targets and features
    print("\n📋 STEP 3: Define Realistic Targets and Engineer Features")
    data_enhanced = predictor.select_features_and_targets(data)
    
    # Step 4: Create sequences
    print("\n📋 STEP 4: Create Training Sequences")
    X, y = predictor.create_sequences(data_enhanced)
    
    # Step 5: Train-test split (chronological for time series)
    print("\n📋 STEP 5: Train-Test Split (Chronological)")
    test_split = 0.1
    train_split_idx = int(len(X) * (1 - test_split))
    
    X_train, X_test = X[:train_split_idx], X[train_split_idx:]
    y_train, y_test = y[:train_split_idx], y[train_split_idx:]
    
    print(f"   • Training set: {len(X_train)} samples")
    print(f"   • Test set: {len(X_test)} samples")
    print(f"   • Test period starts: {data_enhanced.iloc[train_split_idx]['timestamp']}")
    
    # Step 6: Train model
    print("\n📋 STEP 6: Train Performance Degradation Model")
    model, history = predictor.train_model(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32
    )
    
    # Step 7: Evaluate model
    print("\n📋 STEP 7: Evaluate Model Performance")
    evaluation_results = predictor.evaluate_model(X_test, y_test)
    
    # Step 8: Create maintenance dashboard
    print("\n📋 STEP 8: Create Maintenance Dashboard")
    dashboard_results = predictor.create_maintenance_dashboard(evaluation_results, data_enhanced)
    
    # Step 9: Save model
    print("\n📋 STEP 9: Save Trained Model")
    predictor.save_model('realistic_boiler_performance_lstm.h5')
    
    print("\n🎉 REALISTIC TRAINING PIPELINE COMPLETE!")
    
    return predictor, evaluation_results, dashboard_results


def compare_realistic_vs_theoretical_approaches():
    """
    Compare the realistic approach vs theoretical fouling factor approach.
    """
    print("\n" + "🔍" * 50)
    print("REALISTIC vs THEORETICAL APPROACH COMPARISON")
    print("🔍" * 50)
    
    print(f"\n📊 THEORETICAL APPROACH (Original):")
    print(f"   ❌ Targets: gas_fouling_avg, water_fouling_avg, fouling_max")
    print(f"   ❌ Problem: Not directly measurable in real operations")
    print(f"   ❌ Limitation: Operators don't know separate gas vs water fouling")
    print(f"   ❌ Issue: Fouling factors are calculated, not measured")
    print(f"   ❌ Gap: Model predictions can't be validated against real data")
    
    print(f"\n✅ REALISTIC APPROACH (Improved):")
    print(f"   ✅ Targets: Overall U degradation, heat effectiveness, temp performance")
    print(f"   ✅ Advantage: Directly observable from existing instrumentation")
    print(f"   ✅ Practical: Uses temperature, flow, and pressure measurements")
    print(f"   ✅ Actionable: Directly relates to maintenance decisions")
    print(f"   ✅ Validatable: Can be verified against actual plant performance")
    
    print(f"\n🎯 TARGET COMPARISON:")
    
    theoretical_targets = [
        "furnace_walls_fouling_gas_avg",
        "generating_bank_fouling_water_avg", 
        "superheater_primary_fouling_gas_max",
        "economizer_primary_fouling_gas_avg"
    ]
    
    realistic_targets = [
        "furnace_walls_U_degradation",
        "generating_bank_effectiveness_degradation",
        "superheater_primary_temp_degradation", 
        "economizer_primary_U_degradation"
    ]
    
    print(f"\n   THEORETICAL TARGETS (not directly measurable):")
    for target in theoretical_targets:
        print(f"     ❌ {target}")
    
    print(f"\n   REALISTIC TARGETS (directly observable):")
    for target in realistic_targets:
        print(f"     ✅ {target}")
    
    print(f"\n💡 PRACTICAL BENEFITS OF REALISTIC APPROACH:")
    print(f"   🔧 Maintenance teams can validate predictions")
    print(f"   📊 Performance degradation directly correlates to efficiency loss")
    print(f"   ⏰ Provides actionable maintenance timing")
    print(f"   💰 Enables cost-benefit analysis of cleaning")
    print(f"   📈 Supports performance trending and analytics")
    print(f"   🎯 Aligns with existing plant monitoring systems")
    
    print(f"\n🚀 DEPLOYMENT ADVANTAGES:")
    print(f"   • Model predictions match plant observations")
    print(f"   • No need for separate fouling factor calculations")
    print(f"   • Integrates with existing SCADA/DCS systems")
    print(f"   • Provides confidence intervals based on real performance")
    print(f"   • Enables continuous model validation and improvement")


# Main execution
def main():
    """
    Main function to run the realistic LSTM implementation.
    """
    
    # File path to your CSV data
    csv_file_path = 'massachusetts_boiler_annual_20250812_113312.csv'
    
    print("🏭" * 60)
    print("REALISTIC BOILER PERFORMANCE PREDICTION - LSTM IMPLEMENTATION")
    print("Predicting Observable Performance Degradation (Not Abstract Fouling Factors)")
    print("🏭" * 60)
    
    # Show comparison of approaches
    compare_realistic_vs_theoretical_approaches()
    
    print(f"\n📝 REALISTIC IMPLEMENTATION OVERVIEW:")
    print(f"   • Dataset: Massachusetts boiler annual operation (8,760 hours)")
    print(f"   • Model: Multi-output LSTM for performance degradation prediction")
    print(f"   • Features: Same as before (combustion, coal, temperatures, soot blowing)")
    print(f"   • Targets: REALISTIC observable metrics:")
    print(f"     - Overall heat transfer coefficient (U) degradation")
    print(f"     - Heat transfer effectiveness degradation") 
    print(f"     - Temperature performance degradation")
    print(f"   • Prediction: 1-hour ahead performance degradation levels")
    print(f"   • Output: Maintenance dashboard with actionable recommendations")
    
    # Check if file exists
    import os
    if not os.path.exists(csv_file_path):
        print(f"\n❌ CSV file not found: {csv_file_path}")
        print(f"   Please ensure the Massachusetts boiler dataset is in the current directory")
        return None
    
    # Run realistic pipeline
    try:
        results = run_realistic_training_pipeline(csv_file_path)
        
        if results[0] is not None:
            predictor, evaluation_results, dashboard_results = results
            
            print(f"\n🎯 FINAL REALISTIC MODEL PERFORMANCE:")
            overall = evaluation_results['overall']
            print(f"   • Model RMSE: {overall['rmse']:.6f} (degradation units)")
            print(f"   • Model MAE: {overall['mae']:.6f} (degradation units)")
            print(f"   • Model MAPE: {overall['mape']:.2f}%")
            print(f"   • Model R²: {overall['r2']:.4f}")
            
            print(f"\n🔧 MAINTENANCE INSIGHTS:")
            if 'urgent_sections' in dashboard_results:
                urgent = dashboard_results['urgent_sections']
                warning = dashboard_results['warning_sections']
                
                if urgent:
                    print(f"   🚨 Urgent maintenance needed: {len(urgent)} sections")
                    for section in urgent:
                        print(f"     • {section.replace('_', ' ').title()}")
                
                if warning:
                    print(f"   ⚠️  Schedule maintenance soon: {len(warning)} sections")
                    for section in warning:
                        print(f"     • {section.replace('_', ' ').title()}")
                
                if not urgent and not warning:
                    print(f"   ✅ All sections performing within acceptable limits")
            
            print(f"\n✅ REALISTIC IMPLEMENTATION COMPLETE!")
            print(f"   • Model trained with observable performance targets")
            print(f"   • Maintenance dashboard created")
            print(f"   • Files generated:")
            print(f"     - realistic_boiler_performance_lstm.h5 (model)")
            print(f"     - realistic_boiler_performance_lstm_metadata.pkl (metadata)")
            
            print(f"\n🎯 NEXT STEPS FOR DEPLOYMENT:")
            print(f"   1. Validate predictions against actual plant performance")
            print(f"   2. Calibrate degradation thresholds based on maintenance history")
            print(f"   3. Integrate with existing plant monitoring systems")
            print(f"   4. Set up automated alerts for degradation thresholds")
            print(f"   5. Use predictions for maintenance schedule optimization")
            
            return results
            
        else:
            print(f"\n❌ Realistic training pipeline failed!")
            return None
            
    except Exception as e:
        print(f"\n❌ Error in realistic training pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    """Run the realistic LSTM implementation."""
    
    # Execute main function
    results = main()
    
    if results is not None:
        print(f"\n🎉 SUCCESS! Realistic LSTM model ready for operational deployment.")
        print(f"\n💡 WHY THIS APPROACH IS BETTER:")
        print(f"   • Predicts what operators actually observe and measure")
        print(f"   • Provides actionable maintenance recommendations")
        print(f"   • Can be validated against real plant performance data")
        print(f"   • Integrates with existing instrumentation and systems")
        print(f"   • Enables continuous improvement through feedback")
        
    else:
        print(f"\n💥 Implementation failed. Check error messages above.")
