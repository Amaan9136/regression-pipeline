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
        """Initialize regression algorithms"""
        return {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'ElasticNet': ElasticNet(),
            'Random Forest': RandomForestRegressor(random_state=self.config['random_state']),
            'Gradient Boosting': GradientBoostingRegressor(random_state=self.config['random_state']),
            'Extra Trees': ExtraTreesRegressor(random_state=self.config['random_state']),
            'Decision Tree': DecisionTreeRegressor(random_state=self.config['random_state']),
            'K-Neighbors': KNeighborsRegressor(),
            'Support Vector': SVR()
        }
    
    def _initialize_hyperparameters(self):
        """Initialize hyperparameter grids for tuning"""
        return {
            'Linear Regression': {},
            'Ridge Regression': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'Lasso Regression': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'ElasticNet': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]},
            'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
            'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
            'Extra Trees': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
            'Decision Tree': {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
            'K-Neighbors': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
            'Support Vector': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
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
            target_columns = ['compressive_strength', 'target', 'y', 'strength']
            target_col = None
            
            # Find target column
            for col in target_columns:
                if col in df.columns:
                    target_col = col
                    break
            
            if target_col is None:
                # Use last column as target
                target_col = df.columns[-1]
                print(f"No standard target column found, using last column: {target_col}")
            
            # Encode categorical variables
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col != target_col:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.categorical_encoders[col] = le
            
            # Separate features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Encode target if categorical
            if y.dtype == 'object':
                self.target_encoder = LabelEncoder()
                y = self.target_encoder.fit_transform(y)
            
            print(f"Features shape: {X.shape}")
            print(f"Target shape: {y.shape}")
            print(f"Feature columns: {list(X.columns)}")
            
            return X, y, df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None, None, None
    
    def _scale_features(self, X_train, X_test):
        """Scale features based on configuration"""
        if self.config['scaling_method'] == 'standard':
            self.scaler = StandardScaler()
        elif self.config['scaling_method'] == 'robust':
            self.scaler = RobustScaler()
        elif self.config['scaling_method'] == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()  # Default fallback
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def _select_features(self, X_train, y_train, X_test):
        """Select best features if configured"""
        if self.config['feature_selection_k'] == 'all':
            return X_train, X_test
        
        try:
            k = int(self.config['feature_selection_k'])
            if k >= X_train.shape[1]:
                return X_train, X_test
            
            self.feature_selector = SelectKBest(score_func=f_regression, k=k)
            X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
            X_test_selected = self.feature_selector.transform(X_test)
            
            print(f"Selected {k} best features out of {X_train.shape[1]}")
            return X_train_selected, X_test_selected
            
        except (ValueError, TypeError):
            print(f"Invalid feature_selection_k: {self.config['feature_selection_k']}, using all features")
            return X_train, X_test
    
    def train_models(self, X, y, progress_callback=None):
        """Train all models and evaluate performance"""
        print("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'], 
            random_state=self.config['random_state']
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = self._scale_features(X_train, X_test)
        
        # Feature selection
        X_train_final, X_test_final = self._select_features(X_train_scaled, y_train, X_test_scaled)
        
        total_models = len(self.algorithms)
        
        for idx, (name, algorithm) in enumerate(self.algorithms.items()):
            try:
                if progress_callback:
                    progress = 15 + (idx / total_models) * 75  # 15% to 90%
                    progress_callback(f"Training {name}...", progress)
                
                print(f"Training {name}...")
                
                # Hyperparameter tuning if enabled
                if self.config['hyperparameter_tuning'] and name in self.hyperparameters:
                    param_grid = self.hyperparameters[name]
                    if param_grid:  # Only if there are parameters to tune
                        grid_search = GridSearchCV(
                            algorithm, param_grid, 
                            cv=self.config['cross_validation_folds'],
                            scoring='r2',
                            n_jobs=-1
                        )
                        grid_search.fit(X_train_final, y_train)
                        best_model = grid_search.best_estimator_
                        print(f"  Best parameters: {grid_search.best_params_}")
                    else:
                        best_model = algorithm
                        best_model.fit(X_train_final, y_train)
                else:
                    best_model = algorithm
                    best_model.fit(X_train_final, y_train)
                
                # Store trained model
                self.trained_models[name] = best_model
                
                # Make predictions
                y_pred = best_model.predict(X_test_final)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    best_model, X_train_final, y_train, 
                    cv=self.config['cross_validation_folds'], 
                    scoring='r2'
                )
                metrics['cv_mean'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()
                
                self.results[name] = metrics
                
                # Track best model
                if metrics['r2'] > self.best_score:
                    self.best_score = metrics['r2']
                    self.best_model = best_model
                
                print(f"  R² Score: {metrics['r2']:.4f}")
                print(f"  RMSE: {metrics['rmse']:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        print(f"\nBest model: {self._get_best_model_name()} (R² = {self.best_score:.4f})")
        
        # Save models
        self._save_models()
        return self.results

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        return {
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred)
        }
    
    def _get_best_model_name(self):
        """Get the name of the best performing model"""
        if not self.results:
            return "None"
        
        best_name = max(self.results.keys(), key=lambda x: self.results[x]['r2'])
        return best_name
    
    def get_model_summary(self):
        """Get summary of all model results"""
        return self.results.copy()
    
    def _save_models(self):
        """Save trained models to disk"""
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        for name, model in self.trained_models.items():
            try:
                model_data = {
                    'model': model,
                    'scaler': self.scaler,
                    'feature_selector': self.feature_selector,
                    'categorical_encoders': self.categorical_encoders,
                    'target_encoder': self.target_encoder,
                    'feature_names': list(self.data_info.get('columns', [])),
                    'metrics': self.results.get(name, {}),
                    'config': self.config,
                    'timestamp': datetime.now().isoformat()
                }
                
                filename = f"{name.replace(' ', '_').lower()}.pkl"
                filepath = os.path.join(models_dir, filename)
                
                with open(filepath, 'wb') as f:
                    pickle.dump(model_data, f)
                
                print(f"Saved {name} model to {filepath}")
                
            except Exception as e:
                print(f"Error saving {name} model: {str(e)}")
    
    def save_pipeline(self, filepath):
        """Save entire pipeline"""
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
            
            if plot_type == 'comparison' and self.results:
                # Model performance comparison
                models = list(self.results.keys())
                r2_scores = [self.results[model]['r2'] for model in models]
                rmse_scores = [self.results[model]['rmse'] for model in models]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # R² scores
                bars1 = ax1.bar(models, r2_scores, color='skyblue', alpha=0.7)
                ax1.set_title('R² Score Comparison', fontsize=14, fontweight='bold')
                ax1.set_ylabel('R² Score')
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar, score in zip(bars1, r2_scores):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
                
                # RMSE scores
                bars2 = ax2.bar(models, rmse_scores, color='lightcoral', alpha=0.7)
                ax2.set_title('RMSE Comparison', fontsize=14, fontweight='bold')
                ax2.set_ylabel('RMSE')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar, score in zip(bars2, rmse_scores):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_scores)*0.01,
                            f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                
            elif plot_type == 'heatmap' and self.results:
                # Performance metrics heatmap
                metrics_data = []
                models = list(self.results.keys())
                metrics = ['r2', 'mse', 'mae', 'rmse', 'explained_variance']
                
                for model in models:
                    row = []
                    for metric in metrics:
                        if metric in self.results[model]:
                            row.append(self.results[model][metric])
                        else:
                            row.append(0)
                    metrics_data.append(row)
                
                df_heatmap = pd.DataFrame(metrics_data, index=models, columns=metrics)
                
                # Normalize data for better visualization
                df_normalized = df_heatmap.copy()
                for col in df_normalized.columns:
                    if col in ['mse', 'mae', 'rmse']:  # Lower is better
                        df_normalized[col] = 1 - (df_normalized[col] / df_normalized[col].max())
                
                sns.heatmap(df_normalized, annot=True, cmap='RdYlBu_r', 
                           cbar_kws={'label': 'Normalized Performance'}, fmt='.3f')
                plt.title('Model Performance Heatmap', fontsize=16, fontweight='bold')
                plt.ylabel('Models')
                plt.xlabel('Metrics')
                plt.tight_layout()
                
            elif plot_type == 'best_scatter' and self.best_model is not None:
                # Scatter plot for best model (simulated since we don't have test data here)
                # This would ideally use actual predictions vs true values
                plt.text(0.5, 0.5, f'Best Model: {self._get_best_model_name()}\n'
                                   f'R² Score: {self.best_score:.4f}\n\n'
                                   f'Scatter plot would show\nPredicted vs Actual values\n'
                                   f'when test predictions are available',
                        ha='center', va='center', transform=plt.gca().transAxes,
                        fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
                plt.title(f'{self._get_best_model_name()}: Performance Visualization', 
                         fontsize=16, fontweight='bold')
                plt.axis('off')
                
            elif plot_type == 'residuals' and self.best_model is not None:
                # Residuals plot placeholder
                plt.text(0.5, 0.5, f'Residual Analysis\n{self._get_best_model_name()}\n\n'
                                   f'This plot would show:\n'
                                   f'• Residuals vs Predicted values\n'
                                   f'• Distribution of residuals\n'
                                   f'• Model diagnostic information',
                        ha='center', va='center', transform=plt.gca().transAxes,
                        fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))
                plt.title('Residual Analysis', fontsize=16, fontweight='bold')
                plt.axis('off')
                
            elif plot_type == 'feature_importance' and self.best_model is not None:
                # Feature importance plot
                best_model_name = self._get_best_model_name()
                best_model = self.trained_models.get(best_model_name)
                
                if hasattr(best_model, 'feature_importances_'):
                    importances = best_model.feature_importances_
                    feature_names = [f'Feature_{i}' for i in range(len(importances))]
                    
                    # Sort features by importance
                    indices = np.argsort(importances)[::-1]
                    
                    plt.figure(figsize=(12, 8))
                    plt.bar(range(len(importances)), importances[indices], color='forestgreen', alpha=0.7)
                    plt.title(f'Feature Importance - {best_model_name}', fontsize=16, fontweight='bold')
                    plt.xlabel('Features')
                    plt.ylabel('Importance')
                    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
                    plt.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    
                else:
                    plt.text(0.5, 0.5, f'Feature Importance\n{best_model_name}\n\n'
                                       f'Feature importance not available\nfor this model type\n\n'
                                       f'Available for tree-based models:\n'
                                       f'• Random Forest\n• Gradient Boosting\n• Extra Trees\n• Decision Tree',
                            ha='center', va='center', transform=plt.gca().transAxes,
                            fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
                    plt.title('Feature Importance Analysis', fontsize=16, fontweight='bold')
                    plt.axis('off')
            
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
    
    def predict(self, X_new):
        """Make predictions on new data using the best model"""
        if self.best_model is None:
            raise ValueError("No trained model available. Train the pipeline first.")
        
        # Preprocess new data
        X_processed = X_new.copy()
        
        # Apply categorical encoding
        for col, encoder in self.categorical_encoders.items():
            if col in X_processed.columns:
                X_processed[col] = encoder.transform(X_processed[col].astype(str))
        
        # Scale features
        if self.scaler:
            X_processed = self.scaler.transform(X_processed)
        
        # Select features
        if self.feature_selector:
            X_processed = self.feature_selector.transform(X_processed)
        
        # Make predictions
        predictions = self.best_model.predict(X_processed)
        
        # Inverse transform target if needed
        if self.target_encoder:
            predictions = self.target_encoder.inverse_transform(predictions)
        
        return predictions
    
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