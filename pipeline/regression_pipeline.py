import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import warnings
warnings.filterwarnings('ignore')

class RegressionPipeline:
    def __init__(self, config=None):
        """Initialize the regression pipeline with configurable parameters"""
        self.config = self._merge_config(config)
        self.algorithms = self._initialize_algorithms()
        self.hyperparameters = self._initialize_hyperparameters()
        self.results = {}
        self.best_model = None
        self.best_score = -float('inf')
        self.scaler = None
        self.feature_selector = None
        self.trained_models = {}
        self.data_info = {}
        self.categorical_encoders = {}
        self.target_encoder = None
        
    def _merge_config(self, user_config):
        """Merge user config with defaults to ensure all required keys exist"""
        default_config = {
            'test_size': 0.2,
            'random_state': 42,
            'cross_validation_folds': 5,
            'feature_selection_k': 'all',
            'scaling_method': 'standard',
            'hyperparameter_tuning': True,
            'feature_engineering': True,
            'model_evaluation_metrics': ['r2', 'mse', 'mae', 'rmse', 'explained_variance']
        }
        
        if user_config is None:
            return default_config
        
        # Merge configs, user config takes precedence
        merged_config = default_config.copy()
        merged_config.update(user_config)
        
        # Validate scaling_method
        valid_scaling_methods = ['standard', 'robust', 'minmax']
        if merged_config['scaling_method'] not in valid_scaling_methods:
            print(f"Warning: Invalid scaling_method '{merged_config['scaling_method']}'. Using 'standard'.")
            merged_config['scaling_method'] = 'standard'
        
        return merged_config
    
    def _initialize_algorithms(self):
        """Initialize machine learning algorithms"""
        return {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=self.config['random_state']),
            'Lasso Regression': Lasso(random_state=self.config['random_state']),
            'Elastic Net': ElasticNet(random_state=self.config['random_state']),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=self.config['random_state'],
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.config['random_state']
            ),
            'Extra Trees': ExtraTreesRegressor(
                n_estimators=100,
                random_state=self.config['random_state'],
                n_jobs=-1
            ),
            'Support Vector Regression': SVR(),
            'Decision Tree': DecisionTreeRegressor(random_state=self.config['random_state']),
            'K-Nearest Neighbors': KNeighborsRegressor()
        }
    
    def _initialize_hyperparameters(self):
        """Initialize hyperparameter grids for tuning"""
        return {
            'Ridge Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Lasso Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Elastic Net': {
                'alpha': [0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.5, 0.9]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'Support Vector Regression': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            },
            'K-Nearest Neighbors': {
                'n_neighbors': [3, 5, 7, 10],
                'weights': ['uniform', 'distance']
            }
        }
    
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess data from CSV file"""
        try:
            print(f"Loading dataset from: {file_path}")
            
            # Try different encodings to handle various file formats
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"Successfully loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not load file with any encoding")
            
            print(f"Loaded dataset with shape: {df.shape}")
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Store data info
            self.data_info = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict()
            }
            
            # Handle missing values
            print("Handling missing values...")
            missing_info = df.isnull().sum()
            print(f"Missing values per column: {missing_info[missing_info > 0]}")
            
            # Drop rows with >50% missing values
            threshold = len(df.columns) * 0.5
            rows_to_drop = df.isnull().sum(axis=1) > threshold
            df = df[~rows_to_drop]
            print(f"Dropped {rows_to_drop.sum()} rows with >50% missing values")
            
            # Fill remaining missing values
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
            
            print(f"After handling missing values: {df.shape}")
            
            # Separate features and target
            target_column = self._identify_target_column(df)
            print(f"Using '{target_column}' as target column")
            
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Validate target variable for regression
            y = self._validate_and_prepare_target(y)
            
            print(f"Features shape: {X.shape}, Target shape: {y.shape}")
            
            # Apply feature engineering if enabled
            if self.config['feature_engineering']:
                print("Applying feature engineering...")
                X = self._apply_feature_engineering(X)
                print(f"After feature engineering: {X.shape}")
            
            # Encode categorical variables
            X, self.categorical_encoders = self._encode_categorical_variables(X)
            print(f"After encoding: {X.shape}")
            
            # Ensure all features are numeric
            X = self._ensure_numeric_features(X)
            
            print(f"Final data shapes - X: {X.shape}, y: {y.shape}")
            
            return X, y, df
            
        except Exception as e:
            print(f"Error loading/preprocessing data: {str(e)}")
            return None, None, None
    
    def _identify_target_column(self, df):
        """Identify the target column from the dataframe"""
        target_column = None
        
        # Look for common target names
        common_targets = [
            'target', 'label', 'y', 'price', 'value', 'output', 'strength',
            'compressive_strength', 'tensile_strength', 'youngs_modulus'
        ]
        
        for col in common_targets:
            if col in df.columns:
                target_column = col
                break
        
        # If not found, use the last column
        if target_column is None:
            target_column = df.columns[-1]
        
        return target_column
    
    def _validate_and_prepare_target(self, y):
        """Validate and prepare target variable for regression"""
        # Handle string targets (convert to numeric if possible)
        if y.dtype == 'object':
            print("Target variable is categorical, attempting to convert to numeric...")
            try:
                # Try to convert to numeric
                y_numeric = pd.to_numeric(y, errors='coerce')
                
                # If conversion successful (no NaN values)
                if not y_numeric.isnull().any():
                    print("Successfully converted target to numeric")
                    return y_numeric
                else:
                    # Encode categorical target
                    print("Encoding categorical target variable...")
                    self.target_encoder = LabelEncoder()
                    y_encoded = self.target_encoder.fit_transform(y.astype(str))
                    print(f"Encoded {len(self.target_encoder.classes_)} categories")
                    return pd.Series(y_encoded, index=y.index)
            except:
                # Fallback to label encoding
                print("Encoding categorical target variable...")
                self.target_encoder = LabelEncoder()
                y_encoded = self.target_encoder.fit_transform(y.astype(str))
                print(f"Encoded {len(self.target_encoder.classes_)} categories")
                return pd.Series(y_encoded, index=y.index)
        
        # Convert to float to ensure consistency
        return y.astype(float)
    
    def _ensure_numeric_features(self, X):
        """Ensure all features are numeric"""
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    X[col].fillna(X[col].median(), inplace=True)
                except:
                    # If conversion fails, use label encoding
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    if col not in self.categorical_encoders:
                        self.categorical_encoders[col] = le
        
        return X.astype(float)
    
    def _apply_feature_engineering(self, X):
        """Apply feature engineering techniques"""
        X_engineered = X.copy()
        
        # Get numerical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Found {len(numerical_cols)} numerical columns for feature engineering")
        
        # Create interaction features for first few numerical columns
        for i, col1 in enumerate(numerical_cols[:3]):
            for col2 in numerical_cols[i+1:3]:
                if col1 != col2:
                    feature_name = f'{col1}_x_{col2}'
                    X_engineered[feature_name] = X_engineered[col1] * X_engineered[col2]
                    print(f"Created interaction feature: {feature_name}")
        
        # Create polynomial features for first few numerical columns
        for col in numerical_cols[:3]:
            if X_engineered[col].var() > 0:  # Only if column has variance
                # Square feature
                feature_name = f'{col}_squared'
                X_engineered[feature_name] = X_engineered[col] ** 2
                
                # Square root feature (handle negative values)
                feature_name = f'{col}_sqrt'
                X_engineered[feature_name] = np.sqrt(np.abs(X_engineered[col]))
                
                print(f"Created polynomial features for: {col}")
        
        # Log transform for highly skewed features
        for col in numerical_cols:
            try:
                skewness = X_engineered[col].skew()
                if abs(skewness) > 2 and (X_engineered[col] > 0).all():
                    feature_name = f'{col}_log'
                    X_engineered[feature_name] = np.log1p(X_engineered[col])
                    print(f"Created log feature for skewed column {col} (skewness: {skewness:.2f})")
            except:
                continue
        
        return X_engineered
    
    def _encode_categorical_variables(self, X):
        """Encode categorical variables"""
        categorical_encoders = {}
        X_encoded = X.copy()
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        print(f"Found {len(categorical_cols)} categorical columns to encode")
        
        for col in categorical_cols:
            print(f"Encoding categorical column: {col}")
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            categorical_encoders[col] = le
            print(f"  - {col}: {len(le.classes_)} unique categories")
        
        return X_encoded, categorical_encoders
    
    def _setup_preprocessing(self, X):
        """Setup preprocessing pipeline"""
        print(f"Setting up preprocessing with scaling method: {self.config['scaling_method']}")
        
        # Initialize scaler based on config
        if self.config['scaling_method'] == 'standard':
            self.scaler = StandardScaler()
        elif self.config['scaling_method'] == 'robust':
            self.scaler = RobustScaler()
        elif self.config['scaling_method'] == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()  # Default fallback
            print(f"Unknown scaling method '{self.config['scaling_method']}', using StandardScaler")
        
        # Setup feature selection
        if self.config['feature_selection_k'] != 'all':
            k = min(self.config['feature_selection_k'], X.shape[1])
            self.feature_selector = SelectKBest(score_func=f_regression, k=k)
            print(f"Feature selection enabled: selecting top {k} features")
        else:
            self.feature_selector = None
            print("No feature selection applied")
    
    def _preprocess_features(self, X, fit=False, y_train=None):
        """Apply preprocessing to features"""
        if fit:
            X_processed = self.scaler.fit_transform(X)
            if self.feature_selector:
                X_processed = self.feature_selector.fit_transform(X_processed, y_train)
        else:
            X_processed = self.scaler.transform(X)
            if self.feature_selector:
                X_processed = self.feature_selector.transform(X_processed)
        
        return X_processed
    
    def _calculate_metrics(self, y_train, y_pred_train, y_test, y_pred_test):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Training metrics
        metrics['r2_train'] = r2_score(y_train, y_pred_train)
        metrics['mse_train'] = mean_squared_error(y_train, y_pred_train)
        metrics['mae_train'] = mean_absolute_error(y_train, y_pred_train)
        metrics['rmse_train'] = np.sqrt(metrics['mse_train'])
        
        # Test metrics
        metrics['r2_test'] = r2_score(y_test, y_pred_test)
        metrics['mse_test'] = mean_squared_error(y_test, y_pred_test)
        metrics['mae_test'] = mean_absolute_error(y_test, y_pred_test)
        metrics['rmse_test'] = np.sqrt(metrics['mse_test'])
        
        # Overall metrics (using test performance)
        metrics['r2'] = metrics['r2_test']
        metrics['mse'] = metrics['mse_test']
        metrics['mae'] = metrics['mae_test']
        metrics['rmse'] = metrics['rmse_test']
        metrics['explained_variance'] = explained_variance_score(y_test, y_pred_test)
        
        return metrics
    
    def train_models(self, X, y, progress_callback=None):
        """Train all regression models with hyperparameter tuning"""
        print(f"Starting model training with {len(self.algorithms)} algorithms")
        
        # Setup preprocessing
        self._setup_preprocessing(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state']
        )
        
        print(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Preprocessing
        X_train_processed = self._preprocess_features(X_train, fit=True, y_train=y_train)
        X_test_processed = self._preprocess_features(X_test, fit=False)
        
        print(f"After preprocessing - Train: {X_train_processed.shape}, Test: {X_test_processed.shape}")
        
        total_models = len(self.algorithms)
        
        for idx, (name, algorithm) in enumerate(self.algorithms.items()):
            try:
                print(f"\nTraining {name} ({idx+1}/{total_models})...")
                if progress_callback:
                    progress_callback(f"Training {name}...", (idx / total_models) * 80 + 10)
                
                # Hyperparameter tuning if enabled and hyperparameters exist
                if self.config['hyperparameter_tuning'] and name in self.hyperparameters:
                    print(f"  - Performing hyperparameter tuning...")
                    grid_search = GridSearchCV(
                        algorithm, 
                        self.hyperparameters[name],
                        cv=self.config['cross_validation_folds'],
                        scoring='r2',
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(X_train_processed, y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    print(f"  - Best parameters: {best_params}")
                else:
                    print(f"  - Training with default parameters...")
                    best_model = algorithm
                    best_params = {}
                    best_model.fit(X_train_processed, y_train)
                
                # Make predictions
                y_pred_train = best_model.predict(X_train_processed)
                y_pred_test = best_model.predict(X_test_processed)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
                
                # Cross-validation score
                try:
                    cv_scores = cross_val_score(
                        best_model, X_train_processed, y_train, 
                        cv=self.config['cross_validation_folds'], 
                        scoring='r2',
                        n_jobs=-1
                    )
                    metrics['cv_score'] = cv_scores.mean()
                    metrics['cv_std'] = cv_scores.std()
                except:
                    metrics['cv_score'] = np.nan
                    metrics['cv_std'] = np.nan
                
                print(f"  - R² Score: {metrics['r2']:.4f}")
                print(f"  - RMSE: {metrics['rmse']:.4f}")
                if not np.isnan(metrics['cv_score']):
                    print(f"  - CV Score: {metrics['cv_score']:.4f} ± {metrics['cv_std']:.4f}")
                
                # Store results
                self.results[name] = metrics
                self.trained_models[name] = {
                    'model': best_model,
                    'params': best_params,
                    'metrics': metrics
                }
                
                # Update best model
                if metrics['r2'] > self.best_score:
                    self.best_score = metrics['r2']
                    self.best_model = name
                    print(f"  - New best model: {name}")
                
                # Save model
                self._save_model(name, best_model, best_params)
                
            except Exception as e:
                error_msg = f"Error training {name}: {str(e)}"
                print(f"  - {error_msg}")
                if progress_callback:
                    progress_callback(error_msg, (idx / total_models) * 80 + 10)
        
        print(f"\nTraining completed!")
        print(f"Best model: {self.best_model} (R² = {self.best_score:.4f})")
    
    def _save_model(self, name, model, params):
        """Save trained model with metadata"""
        model_data = {
            'model': model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'categorical_encoders': self.categorical_encoders,
            'target_encoder': self.target_encoder,
            'config': self.config,
            'params': params,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            os.makedirs('models', exist_ok=True)
            model_filename = f"models/{name.replace(' ', '_').lower()}_model.pkl"
            with open(model_filename, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"  - Model saved: {model_filename}")
        except Exception as e:
            print(f"  - Error saving model: {str(e)}")
    
    def get_model_summary(self):
        """Get summary of all trained models"""
        return self.results
    
    def get_best_model(self):
        """Get the best performing model"""
        if self.best_model and self.best_model in self.trained_models:
            return self.trained_models[self.best_model]
        return None
    
    def predict(self, X, model_name=None):
        """Make predictions using a specific model or the best model"""
        if model_name is None:
            model_name = self.best_model
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.trained_models[model_name]['model']
        
        # Preprocess features
        X_processed = self._preprocess_features(X, fit=False)
        
        # Make predictions
        predictions = model.predict(X_processed)
        
        # Inverse transform if target was encoded
        if self.target_encoder:
            predictions = self.target_encoder.inverse_transform(predictions.astype(int))
        
        return predictions
    
    def save_pipeline(self, filepath):
        """Save the entire pipeline"""
        pipeline_data = {
            'config': self.config,
            'results': self.results,
            'best_model': self.best_model,
            'best_score': self.best_score,
            'data_info': self.data_info,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'categorical_encoders': self.categorical_encoders,
            'target_encoder': self.target_encoder,
            'trained_models': self.trained_models,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(pipeline_data, f)
            print(f"Pipeline saved to: {filepath}")
        except Exception as e:
            print(f"Error saving pipeline: {str(e)}")
    
    def generate_plots(self, dataset_name, progress_callback=None):
        """Generate comprehensive visualizations"""
        print("Generating visualization plots...")
        plots = {}
        
        plot_configs = [
            ('comparison', 'Model Performance Comparison'),
            ('best_scatter', 'Best Model: Predicted vs Actual'),
            ('heatmap', 'Performance Metrics Heatmap'),
            ('residuals', 'Residual Analysis'),
            ('feature_importance', 'Feature Importance')
        ]
        
        for idx, (plot_type, title) in enumerate(plot_configs):
            try:
                if progress_callback:
                    progress_callback(f"Generating {title}...", 90 + (idx / len(plot_configs)) * 10)
                
                plot_path = self._create_plot(plot_type, title, dataset_name)
                if plot_path:
                    plots[plot_type] = plot_path
                    print(f"  - Generated: {plot_type}")
                    
            except Exception as e:
                print(f"  - Error generating {plot_type} plot: {str(e)}")
        
        return plots
    
    def _create_plot(self, plot_type, title, dataset_name):
        """Create individual plots"""
        try:
            plt.figure(figsize=(12, 8))
            
            if plot_type == 'comparison':
                # Model comparison bar plot
                models = list(self.results.keys())
                r2_scores = [self.results[model]['r2'] for model in models]
                
                colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
                bars = plt.bar(models, r2_scores, color=colors)
                
                plt.title('Model Performance Comparison (R² Score)', fontsize=16, fontweight='bold')
                plt.ylabel('R² Score', fontsize=12)
                plt.xlabel('Models', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, score in zip(bars, r2_scores):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
                
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                
            elif plot_type == 'heatmap':
                # Performance metrics heatmap
                metrics_data = []
                model_names = []
                
                for model in self.results:
                    metrics_data.append([
                        self.results[model]['r2'],
                        self.results[model]['mse'],
                        self.results[model]['mae'],
                        self.results[model]['rmse']
                    ])
                    model_names.append(model)
                
                df_metrics = pd.DataFrame(
                    metrics_data,
                    index=model_names,
                    columns=['R²', 'MSE', 'MAE', 'RMSE']
                )
                
                sns.heatmap(df_metrics, annot=True, cmap='viridis', fmt='.3f', 
                           cbar_kws={'label': 'Metric Value'})
                plt.title('Performance Metrics Heatmap', fontsize=16, fontweight='bold')
                plt.ylabel('Models', fontsize=12)
                plt.xlabel('Metrics', fontsize=12)
                plt.tight_layout()
            
            else:
                # For other plot types, create informational plots
                plt.text(0.5, 0.5, f'{title}\n(Available after training completion)', 
                        ha='center', va='center', transform=plt.gca().transAxes,
                        fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                plt.title(title, fontsize=16, fontweight='bold')
                plt.axis('off')
            
            # Save plot
            os.makedirs('static/plots', exist_ok=True)
            plot_filename = f"{dataset_name}_{plot_type}.png"
            plot_path = f"static/plots/{plot_filename}"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return f"/static/plots/{plot_filename}"
            
        except Exception as e:
            print(f"Error creating {plot_type} plot: {str(e)}")
            plt.close()
            return None
              
    @classmethod
    def load_pipeline(cls, filepath):
        """Load a saved pipeline"""
        try:
            with open(filepath, 'rb') as f:
                pipeline_data = pickle.load(f)
            
            pipeline = cls(pipeline_data['config'])
            pipeline.results = pipeline_data['results']
            pipeline.best_model = pipeline_data['best_model']
            pipeline.best_score = pipeline_data['best_score']
            pipeline.data_info = pipeline_data['data_info']
            pipeline.scaler = pipeline_data['scaler']
            pipeline.feature_selector = pipeline_data['feature_selector']
            pipeline.categorical_encoders = pipeline_data.get('categorical_encoders', {})
            pipeline.target_encoder = pipeline_data.get('target_encoder', None)
            pipeline.trained_models = pipeline_data['trained_models']
            
            print(f"Pipeline loaded from: {filepath}")
            return pipeline
            
        except Exception as e:
            print(f"Error loading pipeline: {str(e)}")
            raise