from flask import Flask, render_template, request, jsonify, send_file, Response
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Create necessary directories
for dir_name in ['data', 'models', 'static/plots']:
    os.makedirs(dir_name, exist_ok=True)

class RegressionPipeline:
    def __init__(self):
        self.algorithms = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Elastic Net': ElasticNet(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVM': SVR(kernel='rbf'),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'KNN': KNeighborsRegressor(n_neighbors=5)
        }
        self.results = {}
        self.best_model = None
        self.best_score = -float('inf')
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the dataset"""
        try:
            df = pd.read_csv(file_path).dropna()
            X, y = df.iloc[:, :-1], df.iloc[:, -1]
            
            # Handle categorical variables
            le_dict = {}
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    le_dict[col] = le
            
            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)
            
            return X, y, df
        except Exception as e:
            return None, None, None
    
    def train_models(self, X, y, progress_callback=None):
        """Train all regression models with progress updates"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.results = {}
        total_models = len(self.algorithms)
        
        for idx, (name, algorithm) in enumerate(self.algorithms.items()):
            try:
                if progress_callback:
                    progress_callback(f"Training {name}...", (idx / total_models) * 80 + 10)
                
                model = algorithm.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Calculate all metrics at once
                mse = mean_squared_error(y_test, y_pred)
                self.results[name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mse),
                    'y_test': y_test,
                    'y_pred': y_pred
                }
                
                # Update best model
                if self.results[name]['r2'] > self.best_score:
                    self.best_score = self.results[name]['r2']
                    self.best_model = name
                
                # Save model
                with open(f"models/{name.replace(' ', '_').lower()}_model.pkl", 'wb') as f:
                    pickle.dump({'model': model, 'scaler': self.scaler}, f)
                
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Error training {name}: {str(e)}", (idx / total_models) * 80 + 10)
                continue
        
        return X_test, y_test
    
    def generate_plots(self, dataset_name, progress_callback=None):
        """Generate visualization plots with progress updates"""
        plots = {}
        plot_steps = [
            ("Generating model comparison chart...", "comparison"),
            ("Creating prediction scatter plot...", "best_scatter"),
            ("Building metrics heatmap...", "heatmap")
        ]
        
        for idx, (message, plot_type) in enumerate(plot_steps):
            if progress_callback:
                progress_callback(message, 90 + (idx / len(plot_steps)) * 10)
            
            plt.figure(figsize=(12, 6) if plot_type == "comparison" else (10, 8))
            
            if plot_type == "comparison":
                models = list(self.results.keys())
                r2_scores = [self.results[model]['r2'] for model in models]
                bars = plt.bar(models, r2_scores, color='skyblue', edgecolor='navy', alpha=0.7)
                plt.title('Model Performance Comparison (R² Score)', fontsize=16, fontweight='bold')
                plt.xlabel('Models', fontsize=12)
                plt.ylabel('R² Score', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', alpha=0.3)
                
                for bar, score in zip(bars, r2_scores):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            elif plot_type == "best_scatter" and self.best_model:
                best_result = self.results[self.best_model]
                plt.scatter(best_result['y_test'], best_result['y_pred'], 
                           alpha=0.6, color='coral', s=50, edgecolors='darkred')
                
                min_val = min(min(best_result['y_test']), min(best_result['y_pred']))
                max_val = max(max(best_result['y_test']), max(best_result['y_pred']))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
                
                plt.xlabel('Actual Values', fontsize=12)
                plt.ylabel('Predicted Values', fontsize=12)
                plt.title(f'Best Model ({self.best_model}) - Predicted vs Actual', fontsize=16, fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            elif plot_type == "heatmap":
                metrics_data = [[self.results[model]['r2'], -self.results[model]['mse'], 
                               -self.results[model]['mae'], -self.results[model]['rmse']] 
                              for model in self.results.keys()]
                
                metrics_df = pd.DataFrame(metrics_data, 
                                         index=list(self.results.keys()),
                                         columns=['R²', 'MSE (neg)', 'MAE (neg)', 'RMSE (neg)'])
                
                sns.heatmap(metrics_df, annot=True, cmap='RdYlBu_r', center=0, 
                           fmt='.3f', square=True, cbar_kws={'label': 'Score'})
                plt.title('Model Performance Metrics Heatmap', fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plot_path = f'static/plots/{plot_type}_{dataset_name}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots[plot_type] = plot_path
        
        return plots

pipeline = RegressionPipeline()

@app.route('/')
def index():
    datasets = [f for f in os.listdir('data') if f.endswith('.csv')] if os.path.exists('data') else []
    return render_template('index.html', datasets=datasets)

@app.route('/train_stream', methods=['POST'])
def train_stream():
    def generate_progress():
        try:
            dataset = request.form['dataset']
            dataset_path = os.path.join('data', dataset)
            
            if not os.path.exists(dataset_path):
                yield f"data: {json.dumps({'error': 'Dataset not found'})}\n\n"
                return
            
            yield f"data: {json.dumps({'progress': 0, 'message': 'Loading dataset...'})}\n\n"
            
            # Load and preprocess data
            X, y, df = pipeline.load_and_preprocess_data(dataset_path)
            if X is None:
                yield f"data: {json.dumps({'error': 'Failed to load dataset'})}\n\n"
                return
            
            yield f"data: {json.dumps({'progress': 5, 'message': 'Dataset loaded successfully'})}\n\n"
            
            # Train models with progress updates
            def progress_callback(message, progress):
                yield f"data: {json.dumps({'progress': int(progress), 'message': message})}\n\n"
            
            # Create a generator that yields progress updates
            def train_with_progress():
                for update in progress_callback("Starting model training...", 10):
                    yield update
                
                pipeline.train_models(X, y, lambda msg, prog: progress_callback(msg, prog).__next__())
                
                for update in progress_callback("Generating visualizations...", 90):
                    yield update
                
                dataset_name = dataset.replace('.csv', '')
                plots = pipeline.generate_plots(dataset_name, lambda msg, prog: progress_callback(msg, prog).__next__())
                
                # Prepare final results
                results_summary = {name: {k: float(v) if k != 'model' and not k.startswith('y_') else v 
                                        for k, v in result.items() if k not in ['model', 'y_test', 'y_pred']}
                                 for name, result in pipeline.results.items()}
                
                response = {
                    'success': True,
                    'results': results_summary,
                    'best_model': pipeline.best_model,
                    'best_score': float(pipeline.best_score),
                    'plots': plots,
                    'dataset_info': {
                        'shape': df.shape,
                        'features': list(X.columns),
                        'target': df.columns[-1]
                    }
                }
                
                yield f"data: {json.dumps({'progress': 100, 'message': 'Training completed!', 'results': response})}\n\n"
            
            # Execute training with progress updates
            yield from train_with_progress()
            
        except Exception as e:
            yield f"data: {json.dumps({'error': f'Training failed: {str(e)}'})}\n\n"
    
    return Response(generate_progress(), mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive'})

@app.route('/train', methods=['POST'])
def train():
    """Legacy endpoint for backward compatibility"""
    try:
        dataset = request.form['dataset']
        dataset_path = os.path.join('data', dataset)
        
        if not os.path.exists(dataset_path):
            return jsonify({'error': 'Dataset not found'})
        
        X, y, df = pipeline.load_and_preprocess_data(dataset_path)
        if X is None:
            return jsonify({'error': 'Failed to load dataset'})
        
        pipeline.train_models(X, y)
        dataset_name = dataset.replace('.csv', '')
        plots = pipeline.generate_plots(dataset_name)
        
        results_summary = {name: {k: float(v) for k, v in result.items() 
                                if k not in ['model', 'y_test', 'y_pred']}
                         for name, result in pipeline.results.items()}
        
        return jsonify({
            'success': True,
            'results': results_summary,
            'best_model': pipeline.best_model,
            'best_score': float(pipeline.best_score),
            'plots': plots,
            'dataset_info': {
                'shape': df.shape,
                'features': list(X.columns),
                'target': df.columns[-1]
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'})

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if not file.filename or not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are allowed'})
        
        file.save(os.path.join('data', file.filename))
        return jsonify({'success': True, 'message': 'File uploaded successfully'})
            
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)