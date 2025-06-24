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
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

class AdvancedRegressionPipeline:
    def __init__(self, config=None):
        """Initialize the advanced regression pipeline with configurable parameters"""
        self.config = config or self._default_config()
        self.algorithms = self._initialize_algorithms()
        self.hyperparameters = self._initialize_hyperparameters()
        self.results = {}
        self.best_model = None
        self.best_score = -float('inf')
        self.scaler = None
        self.feature_selector = None
        self.preprocessing_steps = []
        self.trained_models = {}
        
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
            'Random Forest': RandomForestRegressor(random_state=self.config['random_state']),
            'Extra Trees': ExtraTreesRegressor(random_state=self.config['random_state']),
            'Gradient Boosting': GradientBoostingRegressor(random_state=self.config['random_state']),
            'SVM': SVR(),
            'Decision Tree': DecisionTreeRegressor(random_state=self.config['random_state']),
            'KNN': KNeighborsRegressor(),
            'Neural Network': MLPRegressor(random_state=self.config['random_state'], max_iter=1000)
        }
    
    def _initialize_hyperparameters(self):
        """Initialize hyperparameter grids for tuning"""
        return {
            'Ridge Regression': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'Lasso Regression': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'Elastic Net': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]},
            'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
            'Extra Trees': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
            'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
            'SVM': {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto']},
            'Decision Tree': {'max_depth': [None, 5, 10, 20]},
            'KNN': {'n_neighbors': [3, 5, 7, 9]},
            'Neural Network': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001]}
        }

    def load_and_preprocess_data(self, file_path, target_column=None):
        """Load and preprocess the dataset with advanced feature engineering"""
        try:
            df = pd.read_csv(file_path)
            
            # Handle missing values
            initial_shape = df.shape
            df = self._handle_missing_values(df)
            
            # Separate features and target
            if target_column:
                if target_column not in df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in dataset")
                y = df[target_column]
                X = df.drop(columns=[target_column])
            else:
                # Assume last column is target
                X, y = df.iloc[:, :-1], df.iloc[:, -1]
            
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
                X = self._engineer_features(X)
                
            # Handle categorical variables
            X, self.categorical_encoders = self._encode_categorical_variables(X)
            
            # Handle target variable encoding if categorical
            if y.dtype == 'object':
                self.target_encoder = LabelEncoder()
                y = self.target_encoder.fit_transform(y)
            else:
                self.target_encoder = None
            
            return X, y, df
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Record missing value statistics
        self.missing_stats = df.isnull().sum()
        
        # Drop rows with too many missing values (>50%)
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=threshold)
        
        # Fill remaining missing values
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        return df
    
    def _engineer_features(self, X):
        """Perform feature engineering"""
        X_engineered = X.copy()
        
        # Create interaction features for numerical columns
        numerical_cols = X_engineered.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) >= 2:
            for i in range(min(3, len(numerical_cols))):  # Limit to avoid explosion
                for j in range(i+1, min(3, len(numerical_cols))):
                    col1, col2 = numerical_cols[i], numerical_cols[j]
                    X_engineered[f'{col1}_x_{col2}'] = X_engineered[col1] * X_engineered[col2]
        
        # Create polynomial features for first few numerical columns
        for col in numerical_cols[:3]:  # Limit to first 3 columns
            if X_engineered[col].var() > 0:  # Only if column has variance
                X_engineered[f'{col}_squared'] = X_engineered[col] ** 2
                X_engineered[f'{col}_sqrt'] = np.sqrt(np.abs(X_engineered[col]))
        
        # Log transform for highly skewed features
        for col in numerical_cols:
            if X_engineered[col].skew() > 2 and (X_engineered[col] > 0).all():
                X_engineered[f'{col}_log'] = np.log1p(X_engineered[col])
        
        return X_engineered
    
    def _encode_categorical_variables(self, X):
        """Encode categorical variables"""
        categorical_encoders = {}
        X_encoded = X.copy()
        
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                categorical_encoders[col] = le
        
        return X_encoded, categorical_encoders
    
    def _setup_preprocessing(self, X):
        """Setup preprocessing pipeline"""
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
    
    def train_models(self, X, y, progress_callback=None):
        """Train all regression models with hyperparameter tuning"""
        # Setup preprocessing
        self._setup_preprocessing(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state']
        )
        
        # Preprocessing
        X_train_processed = self._preprocess_features(X_train, fit=True)
        X_test_processed = self._preprocess_features(X_test, fit=False)
        
        self.results = {}
        total_models = len(self.algorithms)
        
        for idx, (name, algorithm) in enumerate(self.algorithms.items()):
            try:
                if progress_callback:
                    progress_callback(f"Training {name}...", (idx / total_models) * 80 + 10)
                
                # Hyperparameter tuning
                if self.config['hyperparameter_tuning'] and name in self.hyperparameters:
                    grid_search = GridSearchCV(
                        algorithm, self.hyperparameters[name],
                        cv=self.config['cross_validation_folds'],
                        scoring='r2', n_jobs=-1
                    )
                    model = grid_search.fit(X_train_processed, y_train)
                    best_model = model.best_estimator_
                    best_params = model.best_params_
                else:
                    best_model = algorithm.fit(X_train_processed, y_train)
                    best_params = {}
                
                # Predictions
                y_pred = best_model.predict(X_test_processed)
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    best_model, X_train_processed, y_train,
                    cv=self.config['cross_validation_folds'], scoring='r2'
                )
                
                # Calculate all metrics
                mse = mean_squared_error(y_test, y_pred)
                self.results[name] = {
                    'model': best_model,
                    'best_params': best_params,
                    'mse': mse,
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mse),
                    'explained_variance': explained_variance_score(y_test, y_pred),
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_test': y_test,
                    'y_pred': y_pred
                }
                
                # Update best model
                if self.results[name]['r2'] > self.best_score:
                    self.best_score = self.results[name]['r2']
                    self.best_model = name
                
                # Save model
                self._save_model(name, best_model, best_params)
                
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Error training {name}: {str(e)}", (idx / total_models) * 80 + 10)
                print(f"Error training {name}: {str(e)}")
    
    def _preprocess_features(self, X, fit=False):
        """Apply preprocessing to features"""
        if fit:
            X_processed = self.scaler.fit_transform(X)
            if self.feature_selector:
                X_processed = self.feature_selector.fit_transform(X_processed, y=None)
        else:
            X_processed = self.scaler.transform(X)
            if self.feature_selector:
                X_processed = self.feature_selector.transform(X_processed)
        
        return X_processed
    
    def _save_model(self, name, model, params):
        """Save trained model with metadata"""
        model_data = {
            'model': model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'categorical_encoders': getattr(self, 'categorical_encoders', {}),
            'target_encoder': getattr(self, 'target_encoder', None),
            'best_params': params,
            'config': self.config,
            'data_info': getattr(self, 'data_info', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"models/{name.replace(' ', '_').lower()}_model.pkl"
        os.makedirs('models', exist_ok=True)
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
    
    def generate_advanced_plots(self, dataset_name, progress_callback=None):
        """Generate comprehensive visualizations"""
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
                    progress_callback(f"Generating {title}...", 90 + (idx / len(plot_configs)) * 8)
                
                plt.figure(figsize=(12, 8))
                
                if plot_type == "comparison":
                    self._plot_model_comparison()
                elif plot_type == "best_scatter":
                    self._plot_best_model_scatter()
                elif plot_type == "heatmap":
                    self._plot_metrics_heatmap()
                elif plot_type == "residuals":
                    self._plot_residual_analysis()
                elif plot_type == "feature_importance":
                    self._plot_feature_importance()
                
                plt.tight_layout()
                plot_path = f'static/plots/{plot_type}_{dataset_name}.png'
                os.makedirs('static/plots', exist_ok=True)
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots[plot_type] = plot_path
                
            except Exception as e:
                print(f"Error generating {plot_type} plot: {str(e)}")
        
        return plots
    
    def _plot_model_comparison(self):
        """Plot model performance comparison"""
        models = list(self.results.keys())
        r2_scores = [self.results[model]['r2'] for model in models]
        colors = ['#FF6B6B' if model == self.best_model else '#4ECDC4' for model in models]
        
        bars = plt.bar(models, r2_scores, color=colors, alpha=0.8)
        plt.title('Model Performance Comparison (R² Score)', fontsize=16, fontweight='bold')
        plt.ylabel('R² Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_best_model_scatter(self):
        """Plot best model predictions vs actual"""
        if self.best_model and self.best_model in self.results:
            result = self.results[self.best_model]
            y_test, y_pred = result['y_test'], result['y_pred']
            
            plt.scatter(y_test, y_pred, alpha=0.6, color='#4ECDC4', s=50)
            
            # Perfect prediction line
            min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            plt.xlabel('Actual Values', fontsize=12)
            plt.ylabel('Predicted Values', fontsize=12)
            plt.title(f'Best Model ({self.best_model}): Predicted vs Actual\nR² = {result["r2"]:.4f}', 
                     fontsize=16, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    def _plot_metrics_heatmap(self):
        """Plot performance metrics heatmap"""
        metrics = ['r2', 'mse', 'mae', 'rmse', 'explained_variance']
        data = []
        
        for model in self.results.keys():
            row = []
            for metric in metrics:
                value = self.results[model][metric]
                # Normalize MSE, MAE, RMSE (lower is better) by taking negative
                if metric in ['mse', 'mae', 'rmse']:
                    value = -value
                row.append(value)
            data.append(row)
        
        df_metrics = pd.DataFrame(data, index=list(self.results.keys()), columns=metrics)
        
        sns.heatmap(df_metrics, annot=True, cmap='RdYlBu_r', center=0, 
                   fmt='.3f', square=True, cbar_kws={'label': 'Score'})
        plt.title('Model Performance Metrics Heatmap', fontsize=16, fontweight='bold')
    
    def _plot_residual_analysis(self):
        """Plot residual analysis for best model"""
        if self.best_model and self.best_model in self.results:
            result = self.results[self.best_model]
            y_test, y_pred = result['y_test'], result['y_pred']
            residuals = y_test - y_pred
            
            plt.subplot(2, 2, 1)
            plt.scatter(y_pred, residuals, alpha=0.6, color='#4ECDC4')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Predicted')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.hist(residuals, bins=20, alpha=0.7, color='#4ECDC4', edgecolor='black')
            plt.xlabel('Residuals')
            plt.ylabel('Frequency')
            plt.title('Residuals Distribution')
            plt.grid(True, alpha=0.3)
            
            plt.suptitle(f'Residual Analysis - {self.best_model}', fontsize=14, fontweight='bold')
    
    def _plot_feature_importance(self):
        """Plot feature importance for best model if available"""
        if self.best_model and self.best_model in self.results:
            model = self.results[self.best_model]['model']
            
            # Get feature importance based on model type
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_names = [f'Feature_{i}' for i in range(len(importance))]
                
                # Sort by importance
                indices = np.argsort(importance)[::-1][:10]  # Top 10 features
                
                plt.bar(range(len(indices)), importance[indices], color='#4ECDC4', alpha=0.8)
                plt.title(f'Top 10 Feature Importance - {self.best_model}', fontsize=16, fontweight='bold')
                plt.ylabel('Importance Score')
                plt.xlabel('Features')
                plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, f'Feature importance not available\nfor {self.best_model}', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                plt.title('Feature Importance', fontsize=16, fontweight='bold')
    
    def get_model_summary(self):
        """Get comprehensive model summary"""
        summary = {
            'total_models': len(self.results),
            'best_model': self.best_model,
            'best_score': self.best_score,
            'data_info': getattr(self, 'data_info', {}),
            'config': self.config,
            'model_results': {}
        }
        
        for name, result in self.results.items():
            summary['model_results'][name] = {
                'r2': result['r2'],
                'mse': result['mse'],
                'mae': result['mae'],
                'rmse': result['rmse'],
                'explained_variance': result['explained_variance'],
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std'],
                'best_params': result['best_params']
            }
        
        return summary