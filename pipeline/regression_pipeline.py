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
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import warnings
warnings.filterwarnings('ignore')

class RegressionPipeline:
    def __init__(self, config=None):
        """Initialize the regression pipeline with configurable parameters"""
        self.config = config or self._default_config()
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
        
    def _default_config(self):
        """Default configuration for the pipeline"""
        return {
            'test_size': 0.2,
            'random_state': 42,
            'cross_validation_folds': 5,
            'feature_selection_k': 'all',
            'scaling_method': 'standard',
            'hyperparameter_tuning': True,
            'feature_engineering': True,
            'model_evaluation_metrics': ['r2', 'mse', 'mae', 'rmse', 'explained_variance']
        }
    
    def _initialize_algorithms(self):
        """Initialize machine learning algorithms"""
        return {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=self.config['random_state']),
            'Lasso Regression': Lasso(random_state=self.config['random_state']),
            'Elastic Net': ElasticNet(random_state=self.config['random_state']),
            'Random Forest': RandomForestRegressor(
                random_state=self.config['random_state'],
                n_estimators=100
            ),
            'Extra Trees': ExtraTreesRegressor(
                random_state=self.config['random_state'],
                n_estimators=100
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                random_state=self.config['random_state'],
                n_estimators=100
            ),
            'SVM': SVR(kernel='rbf'),
            'Decision Tree': DecisionTreeRegressor(random_state=self.config['random_state']),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'Neural Network': MLPRegressor(
                random_state=self.config['random_state'], 
                max_iter=1000,
                hidden_layer_sizes=(100,)
            )
        }
    
    def _initialize_hyperparameters(self):
        """Initialize hyperparameter grids for tuning"""
        return {
            'Ridge Regression': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'Lasso Regression': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'Elastic Net': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]},
            'Random Forest': {
                'n_estimators': [50, 100, 200], 
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'Extra Trees': {
                'n_estimators': [50, 100, 200], 
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200], 
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'SVM': {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto'], 'epsilon': [0.1, 0.01]},
            'Decision Tree': {'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 5, 10]},
            'KNN': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
            'Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)], 
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }

    def load_and_preprocess_data(self, file_path, target_column=None):
        """Load and preprocess the dataset with feature engineering"""
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded dataset with shape: {df.shape}")
            
            # Handle missing values
            initial_shape = df.shape
            df = self._handle_missing_values(df)
            print(f"After handling missing values: {df.shape}")
            
            # Separate features and target
            if target_column:
                if target_column not in df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in dataset")
                y = df[target_column]
                X = df.drop(columns=[target_column])
            else:
                # Assume last column is target
                X, y = df.iloc[:, :-1], df.iloc[:, -1]
            
            print(f"Features shape: {X.shape}, Target shape: {y.shape}")
            
            # Store original data info
            self.data_info = {
                'original_shape': initial_shape,
                'processed_shape': df.shape,
                'feature_names': list(X.columns),
                'target_name': y.name if hasattr(y, 'name') else 'target',
                'missing_values_handled': initial_shape[0] - df.shape[0]
            }
            
            # Feature engineering
            if self.config['feature_engineering']:
                print("Applying feature engineering...")
                X = self._engineer_features(X)
                print(f"After feature engineering: {X.shape}")
                
            # Handle categorical variables
            X, self.categorical_encoders = self._encode_categorical_variables(X)
            print(f"After encoding: {X.shape}")
            
            # Handle target variable encoding if categorical
            if y.dtype == 'object':
                print("Encoding categorical target variable...")
                self.target_encoder = LabelEncoder()
                y = pd.Series(self.target_encoder.fit_transform(y), index=y.index)
            else:
                self.target_encoder = None
            
            # Final validation
            if X.isnull().any().any() or y.isnull().any():
                print("Warning: Still have missing values after preprocessing")
                X = X.fillna(X.mean() if X.dtypes.name in ['int64', 'float64'] else X.mode().iloc[0])
                y = y.fillna(y.mean() if y.dtype.name in ['int64', 'float64'] else y.mode().iloc[0])
            
            print(f"Final data shapes - X: {X.shape}, y: {y.shape}")
            return X, y, df
            
        except Exception as e:
            print(f"Error in load_and_preprocess_data: {str(e)}")
            raise Exception(f"Error loading data: {str(e)}")
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("Handling missing values...")
        
        # Record missing value statistics
        self.missing_stats = df.isnull().sum()
        print(f"Missing values per column: {self.missing_stats[self.missing_stats > 0]}")
        
        # Drop rows with too many missing values (>50%)
        threshold = len(df.columns) * 0.5
        initial_rows = len(df)
        df = df.dropna(thresh=threshold)
        print(f"Dropped {initial_rows - len(df)} rows with >50% missing values")
        
        # Fill remaining missing values
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ['int64', 'float64']:
                    fill_value = df[col].median()
                    df[col].fillna(fill_value, inplace=True)
                    print(f"Filled {col} with median: {fill_value}")
                else:
                    mode_values = df[col].mode()
                    fill_value = mode_values[0] if not mode_values.empty else 'Unknown'
                    df[col].fillna(fill_value, inplace=True)
                    print(f"Filled {col} with mode: {fill_value}")
        
        return df
    
    def _engineer_features(self, X):
        """Perform feature engineering"""
        X_engineered = X.copy()
        
        # Get numerical columns
        numerical_cols = X_engineered.select_dtypes(include=[np.number]).columns
        print(f"Found {len(numerical_cols)} numerical columns for feature engineering")
        
        if len(numerical_cols) >= 2:
            # Create interaction features for first few numerical columns
            for i in range(min(3, len(numerical_cols))):
                for j in range(i+1, min(3, len(numerical_cols))):
                    col1, col2 = numerical_cols[i], numerical_cols[j]
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
            skewness = X_engineered[col].skew()
            if abs(skewness) > 2 and (X_engineered[col] > 0).all():
                feature_name = f'{col}_log'
                X_engineered[feature_name] = np.log1p(X_engineered[col])
                print(f"Created log feature for skewed column {col} (skewness: {skewness:.2f})")
        
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
            self.scaler = StandardScaler()  # Default
        
        # Setup feature selection
        if self.config['feature_selection_k'] != 'all':
            k = min(self.config['feature_selection_k'], X.shape[1])
            self.feature_selector = SelectKBest(score_func=f_regression, k=k)
            print(f"Feature selection enabled: selecting top {k} features")
        else:
            self.feature_selector = None
            print("No feature selection applied")
    
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
                cv_scores = cross_val_score(
                    best_model, X_train_processed, y_train, 
                    cv=self.config['cross_validation_folds'], 
                    scoring='r2',
                    n_jobs=-1
                )
                metrics['cv_score'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()
                
                print(f"  - R² Score: {metrics['r2']:.4f}")
                print(f"  - RMSE: {metrics['rmse']:.4f}")
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
        
        print(f"\nTraining completed! Best model: {self.best_model} (R² = {self.best_score:.4f})")
    
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
    
    def _save_model(self, name, model, params):
        """Save trained model with metadata"""
        model_data = {
            'model': model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'categorical_encoders': self.categorical_encoders,
            'target_encoder': self.target_encoder,
            'best_params': params,
            'config': self.config,
            'data_info': self.data_info,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs('models', exist_ok=True)
        filename = f"models/{name.replace(' ', '_').lower()}_model.pkl"
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"  - Model saved: {filename}")
        except Exception as e:
            print(f"  - Error saving model: {str(e)}")
    
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