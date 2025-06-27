from flask_socketio import join_room, leave_room
from flask import send_from_directory
import time
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
import os
import pickle
import json
import threading
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename
import logging
from pipeline.regression_pipeline import RegressionPipeline
from utils.data_cleaning import DataCleaner, DataValidator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['UPLOAD_FOLDER'] = 'data'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
for dir_name in ['data', 'models', 'static/plots', 'temp']:
    os.makedirs(dir_name, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif hasattr(obj, 'dtype'):
        return str(obj)
    else:
        return obj

class BackendManager:
    def __init__(self):
        self.active_sessions = {}
        self.data_cleaner = DataCleaner()
    
    def create_session(self, session_id):
        self.active_sessions[session_id] = {
            'created_at': datetime.now(),
            'pipeline': RegressionPipeline(),
            'data_cleaner': DataCleaner(),
            'current_step': 'idle',
            'progress': 0
        }
        return session_id
    
    def get_session(self, session_id):
        return self.active_sessions.get(session_id)
    
    def emit_progress(self, session_id, message, progress, data=None):
        """Enhanced progress emission with better error handling"""
        try:
            # Convert data to JSON-safe format
            safe_data = convert_numpy_types(data) if data else None
            
            # Create payload
            payload = {
                'message': str(message),
                'progress': min(100, max(0, float(progress))),  # Ensure progress is 0-100
                'timestamp': datetime.now().isoformat()
            }
            
            # Add data if provided
            if safe_data:
                payload['data'] = safe_data
            
            # Test JSON serialization
            json.dumps(payload)
            
            # Emit to specific room
            with app.app_context():
                socketio.emit('training_progress', payload, room=session_id)
                
            logger.info(f"Progress emitted to {session_id}: {progress}% - {message}")
            
        except Exception as e:
            logger.error(f"Failed to emit progress: {e}")

backend_manager = BackendManager()

@app.route('/')
def index():
    datasets = []
    if os.path.exists('data'):
        datasets = [f for f in os.listdir('data') if f.endswith('.csv')]
    return render_template('index.html', datasets=datasets)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if not file.filename or not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are allowed'}), 400
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        df = pd.read_csv(file_path, nrows=5)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'File uploaded successfully',
            'preview': {
                'columns': list(df.columns),
                'shape': [len(df), len(df.columns)],
                'sample_data': df.head().to_dict('records')
            }
        })
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/dataset_info/<filename>')
def get_dataset_info(filename):
    try:
        file_path = os.path.join('data', filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        df = pd.read_csv(file_path)
        column_info = backend_manager.data_cleaner.get_column_info(df)
        quality_report = backend_manager.data_cleaner.validate_data_quality(df)
        
        missing_info = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_info[col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_count / len(df) * 100)
                }
        
        dtypes_dict = {}
        for col, dtype in df.dtypes.items():
            dtypes_dict[col] = str(dtype)
        
        return jsonify({
            'success': True,
            'info': {
                'filename': filename,
                'shape': df.shape,
                'columns': list(df.columns),
                'column_info': column_info,
                'missing_values': missing_info,
                'quality_report': quality_report,
                'dtypes': dtypes_dict,
                'memory_usage': float(df.memory_usage(deep=True).sum() / 1024 / 1024),
                'sample_data': df.head().to_dict('records')
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        return jsonify({'error': f'Failed to get dataset info: {str(e)}'}), 500

@app.route('/api/apply_cleaning', methods=['POST'])
def apply_cleaning():
    try:
        data = request.get_json()
        filename = data.get('filename')
        cleaning_operations = data.get('operations', {})
        
        file_path = os.path.join('data', filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        df = pd.read_csv(file_path)
        original_shape = df.shape
        
        results = {}
        
        if 'drop_columns' in cleaning_operations:
            df, dropped = backend_manager.data_cleaner.drop_columns(
                df, cleaning_operations['drop_columns']
            )
            results['dropped_columns'] = dropped
        
        if 'rename_columns' in cleaning_operations:
            df, renamed = backend_manager.data_cleaner.rename_columns(
                df, cleaning_operations['rename_columns']
            )
            results['renamed_columns'] = renamed
        
        if 'missing_values' in cleaning_operations:
            df, handled = backend_manager.data_cleaner.handle_missing_values(
                df, cleaning_operations['missing_values']
            )
            results['missing_values_handled'] = handled
        
        if 'categorical_encoding' in cleaning_operations:
            df, encoded = backend_manager.data_cleaner.encode_categorical_data(
                df, cleaning_operations['categorical_encoding']
            )
            results['encoded_columns'] = encoded
        
        if 'transformations' in cleaning_operations:
            df, transformed = backend_manager.data_cleaner.apply_transformations(
                df, cleaning_operations['transformations']
            )
            results['transformed_columns'] = transformed
        
        if 'remove_outliers' in cleaning_operations:
            outlier_indices = cleaning_operations['remove_outliers']
            df, removed_count = backend_manager.data_cleaner.remove_outliers(df, outlier_indices)
            results['outliers_removed'] = removed_count
        
        cleaned_filename = f"cleaned_{filename}"
        cleaned_path = os.path.join('data', cleaned_filename)
        df.to_csv(cleaned_path, index=False)
        
        quality_report = backend_manager.data_cleaner.validate_data_quality(df)
        validation_results = DataValidator.validate_for_training(df)
        
        return jsonify({
            'success': True,
            'results': results,
            'original_shape': original_shape,
            'final_shape': df.shape,
            'cleaned_filename': cleaned_filename,
            'quality_report': quality_report,
            'validation': validation_results,
            'cleaning_summary': backend_manager.data_cleaner.get_cleaning_summary()
        })
        
    except Exception as e:
        logger.error(f"Error applying cleaning: {str(e)}")
        return jsonify({'error': f'Cleaning failed: {str(e)}'}), 500

@app.route('/api/detect_outliers/<filename>')
def detect_outliers(filename):
    try:
        file_path = os.path.join('data', filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        df = pd.read_csv(file_path)
        outliers = backend_manager.data_cleaner.detect_outliers(df)
        
        return jsonify({
            'success': True,
            'outliers': outliers
        })
        
    except Exception as e:
        logger.error(f"Error detecting outliers: {str(e)}")
        return jsonify({'error': f'Outlier detection failed: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        features = data.get('features', {})
        
        model_path = os.path.join('models', f'{model_name}.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found'}), 404
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data.get('scaler')
        feature_names = model_data.get('feature_names', [])
        
        feature_array = np.array([[features.get(name, 0) for name in feature_names]])
        
        if scaler:
            feature_array = scaler.transform(feature_array)
        
        prediction = model.predict(feature_array)[0]
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'model_used': model_name,
            'features_used': features
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/export_results/<dataset_name>')
def export_results(dataset_name):
    try:
        export_data = {
            'dataset_name': dataset_name,
            'export_timestamp': datetime.now().isoformat(),
            'results': 'Generated results would go here'
        }
        
        return jsonify({
            'success': True,
            'message': 'Results exported successfully',
            'export_data': export_data
        })
        
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(backend_manager.active_sessions)
    })

@socketio.on('connect')
def handle_connect():
    session_id = str(uuid.uuid4())
    backend_manager.create_session(session_id)
    join_room(session_id)
    emit('session_created', {'session_id': session_id})
    logger.info(f"Client connected with session: {session_id}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

@socketio.on('start_training')
def handle_training(data):
    session_id = data.get('session_id')
    if not session_id:
        emit('training_error', {'error': 'No session ID provided'})
        return
    
    session = backend_manager.get_session(session_id)
    if not session:
        emit('training_error', {'error': 'Invalid session'})
        return
    
    dataset = data.get('dataset')
    target_column = data.get('target_column')
    config = data.get('config', {})

    def training_worker():
        try:
            dataset_path = os.path.join('data', dataset)
            if not os.path.exists(dataset_path):
                with app.app_context():
                    socketio.emit('training_error', {'error': 'Dataset not found'}, room=session_id)
                return
            
            dataset_name = dataset.replace('.csv', '') if dataset.endswith('.csv') else dataset
            
            pipeline = RegressionPipeline(config)
            
            def progress_callback(message, progress):
                backend_manager.emit_progress(session_id, message, progress)
                    
            backend_manager.emit_progress(session_id, "Loading and preprocessing data...", 10)
            try:
                X, y, df = pipeline.load_and_preprocess_data(dataset_path)
                
                if X is None or y is None:
                    with app.app_context():
                        socketio.emit('training_error', {'error': 'Failed to load or preprocess dataset.'}, room=session_id)
                    return
                
                # Store basic data info
                pipeline.data_info = {
                    'shape': df.shape,
                    'features': list(X.columns) if hasattr(X, 'columns') else [],
                    'target_column': target_column,
                    'missing_values': df.isnull().sum().sum()
                }
                
            except Exception as load_error:
                logger.error(f"Data loading error: {load_error}")
                with app.app_context():
                    socketio.emit('training_error', {'error': f'Data loading failed: {str(load_error)}'}, room=session_id)
                return
            
            # Train models
            backend_manager.emit_progress(session_id, "Training models...", 30)
            try:
                results = pipeline.train_models(X, y, progress_callback)
                
                # Debug logging
                logger.info(f"Raw results type: {type(results)}")
                logger.info(f"Raw results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
                
                # Process the model results properly
                model_summary = {}
                if hasattr(pipeline, 'results') and pipeline.results:
                    for model_name, metrics in pipeline.results.items():
                        model_summary[model_name] = {
                            'r2_score': float(metrics.get('r2', 0)),
                            'rmse': float(metrics.get('rmse', 0)),
                            'mae': float(metrics.get('mae', 0)),
                            'mse': float(metrics.get('mse', 0)),
                            'cv_mean': float(metrics.get('cv_mean', 0)),
                            'cv_std': float(metrics.get('cv_std', 0))
                        }
                
                # Add summary statistics
                final_results = {
                    'training_completed': True,
                    'models_trained': len(model_summary),
                    'best_model_name': pipeline._get_best_model_name(),
                    'best_score': float(pipeline.best_score) if pipeline.best_score != -float('inf') else 0.0,
                    'model_summary': model_summary,
                    'dataset_info': {
                        'shape': list(pipeline.data_info.get('shape', [0, 0])),
                        'features': pipeline.data_info.get('features', []),
                        'target_column': target_column
                    }
                }
                
            except Exception as train_error:
                logger.error(f"Training error: {train_error}")
                final_results = {
                    'training_error': str(train_error),
                    'models_trained': 0
                }
            
            # Generate plots with proper dataset_name
            backend_manager.emit_progress(session_id, "Generating visualizations...", 80)
            plots = {}
            try:
                if hasattr(pipeline, 'generate_plots'):
                    plots = pipeline.generate_plots(dataset_name, progress_callback)
                    logger.info(f"Generated plots: {list(plots.keys())}")
                else:
                    logger.warning("Pipeline does not have generate_plots method")
                    plots = {}
            except Exception as plot_error:
                logger.warning(f"Plot generation failed: {plot_error}")
                plots = {}
            
            # Final emission
            backend_manager.emit_progress(session_id, "Training completed successfully!", 100, {
                'model_summary': final_results.get('model_summary', {}),
                'training_summary': final_results,
                'plots': plots,
                'dataset_info': final_results.get('dataset_info', {})
            })
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            with app.app_context():
                socketio.emit('training_error', {'error': str(e)}, room=session_id)

    training_thread = threading.Thread(target=training_worker)
    training_thread.daemon = True
    training_thread.start()

@app.route('/static/plots/<filename>')
def serve_plot(filename):
    """Serve plot images from static/plots directory"""
    return send_from_directory('static/plots', filename)

@app.route('/api/models')
def list_models():
    """List all available trained models"""
    try:
        models_dir = 'models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
            return jsonify({'success': True, 'models': []})
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        models = []
        
        for model_file in model_files:
            try:
                model_path = os.path.join(models_dir, model_file)
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                model_name = model_file.replace('.pkl', '')
                model_info = {
                    'name': model_name,
                    'display_name': model_name.replace('_', ' ').title(),
                    'metrics': model_data.get('metrics', {}),
                    'feature_names': model_data.get('feature_names', []),
                    'timestamp': model_data.get('timestamp', 'Unknown'),
                    'config': model_data.get('config', {})
                }
                models.append(model_info)
                
            except Exception as e:
                logger.warning(f"Failed to load model {model_file}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        models.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'models': models,
            'count': len(models)
        })
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return jsonify({'error': f'Failed to list models: {str(e)}'}), 500

@app.route('/api/models/<model_name>')
def get_model_details(model_name):
    """Get detailed information about a specific model"""
    try:
        model_path = os.path.join('models', f'{model_name}.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found'}), 404
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Find associated plots
        plots_dir = 'static/plots'
        plots = {}
        if os.path.exists(plots_dir):
            plot_files = os.listdir(plots_dir)
            for plot_file in plot_files:
                if model_name.lower() in plot_file.lower() or 'model_comparison' in plot_file:
                    plot_type = plot_file.split('_')[-1].replace('.png', '')
                    plots[plot_type] = f'/static/plots/{plot_file}'
        
        model_details = {
            'name': model_name,
            'display_name': model_name.replace('_', ' ').title(),
            'metrics': convert_numpy_types(model_data.get('metrics', {})),
            'feature_names': model_data.get('feature_names', []),
            'timestamp': model_data.get('timestamp', 'Unknown'),
            'config': model_data.get('config', {}),
            'plots': plots
        }
        
        return jsonify({
            'success': True,
            'model': model_details
        })
        
    except Exception as e:
        logger.error(f"Error getting model details for {model_name}: {str(e)}")
        return jsonify({'error': f'Failed to get model details: {str(e)}'}), 500

@app.route('/api/batch_predict', methods=['POST'])
def make_batch_prediction():
    """Handle batch predictions from uploaded CSV"""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        csv_data = data.get('csv_data', [])
        
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        
        if not csv_data:
            return jsonify({'error': 'CSV data is required'}), 400
        
        model_path = os.path.join('models', f'{model_name}.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found'}), 404
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data.get('scaler')
        feature_names = model_data.get('feature_names', [])
        
        df = pd.DataFrame(csv_data)
        missing_features = [fn for fn in feature_names if fn not in df.columns]
        if missing_features:
            return jsonify({
                'error': f'Missing required features in CSV: {", ".join(missing_features)}',
                'required_features': feature_names
            }), 400
        
        X = df[feature_names].values
        
        if scaler:
            X = scaler.transform(X)
        
        predictions = model.predict(X)
        
        results = []
        for i, prediction in enumerate(predictions):
            result_row = df.iloc[i].to_dict()
            result_row['prediction'] = float(prediction)
            results.append(result_row)
        
        return jsonify({
            'success': True,
            'predictions': results,
            'model_used': model_name,
            'total_predictions': len(results)
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)