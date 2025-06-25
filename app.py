from flask import Flask, render_template, request, jsonify, send_file
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
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Initialize SocketIO for real-time communication
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Create necessary directories
for dir_name in ['data', 'models', 'static/plots', 'temp']:
    os.makedirs(dir_name, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'dtype'):
            return str(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

class BackendManager:
    """Manages backend operations and real-time communication"""
    
    def __init__(self):
        self.active_sessions = {}
        self.data_cleaner = DataCleaner()
    
    def create_session(self, session_id):
        """Create a new session for a user"""
        self.active_sessions[session_id] = {
            'created_at': datetime.now(),
            'pipeline': RegressionPipeline(),
            'data_cleaner': DataCleaner(),
            'current_step': 'idle',
            'progress': 0
        }
        return session_id
    
    def get_session(self, session_id):
        """Get session data"""
        return self.active_sessions.get(session_id)
    
    def emit_progress(self, session_id, message, progress, data=None):
        """Emit progress update to specific session using app context"""
        with app.app_context():
            socketio.emit('training_progress', {
                'message': message,
                'progress': progress,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }, room=session_id)

backend_manager = BackendManager()

@app.route('/')
def index():
    """Main dashboard page"""
    datasets = []
    if os.path.exists('data'):
        datasets = [f for f in os.listdir('data') if f.endswith('.csv')]
    return render_template('index.html', datasets=datasets)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and save dataset"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if not file.filename or not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are allowed'}), 400
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Get basic info about the uploaded file
        df = pd.read_csv(file_path, nrows=5)  # Quick preview
        
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
    """Get comprehensive dataset information"""
    try:
        file_path = os.path.join('data', filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        df = pd.read_csv(file_path)
        
        # Get column information
        column_info = backend_manager.data_cleaner.get_column_info(df)
        
        # Get data quality report
        quality_report = backend_manager.data_cleaner.validate_data_quality(df)
        
        # Get missing values info
        missing_info = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_info[col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_count / len(df) * 100)
                }
        
        # Convert dtypes to string to avoid JSON serialization issues
        dtypes_dict = {}
        for col, dtype in df.dtypes.items():
            dtypes_dict[col] = str(dtype)
        
        return jsonify({
            'success': True,
            'dataset_info': {
                'filename': filename,
                'shape': df.shape,
                'columns': column_info,
                'missing_values': missing_info,
                'quality_report': quality_report,
                'dtypes': dtypes_dict,
                'memory_usage': float(df.memory_usage(deep=True).sum() / 1024 / 1024)  # MB
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        return jsonify({'error': f'Failed to get dataset info: {str(e)}'}), 500

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    session_id = str(uuid.uuid4())
    backend_manager.create_session(session_id)
    emit('session_created', {'session_id': session_id})
    logger.info(f"Client connected with session: {session_id}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected")

@socketio.on('start_training')
def handle_training(data):
    """Handle training request with real-time progress updates"""
    session_id = data.get('session_id')
    dataset = data.get('dataset')
    config = data.get('config', {})
    
    # Ensure config has required defaults
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
    
    # Merge user config with defaults
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
    
    session = backend_manager.get_session(session_id)
    if not session:
        with app.app_context():
            emit('training_error', {'error': 'Invalid session'})
        return
    
    def training_worker():
        try:
            dataset_path = os.path.join('data', dataset)
            if not os.path.exists(dataset_path):
                with app.app_context():
                    socketio.emit('training_error', {'error': 'Dataset not found'}, room=session_id)
                return
            
            # Initialize pipeline with custom config
            pipeline = RegressionPipeline(config)
            
            # Progress callback for real-time updates
            def progress_callback(message, progress):
                backend_manager.emit_progress(session_id, message, progress)
            
            # Load and preprocess data
            backend_manager.emit_progress(session_id, "Loading dataset...", 5)
            X, y, df = pipeline.load_and_preprocess_data(dataset_path)
            
            if X is None:
                with app.app_context():
                    socketio.emit('training_error', {'error': 'Failed to load dataset'}, room=session_id)
                return
            
            backend_manager.emit_progress(session_id, "Dataset loaded successfully", 10)
            
            # Train models
            backend_manager.emit_progress(session_id, "Starting model training...", 15)
            pipeline.train_models(X, y, progress_callback)
            
            # Generate plots
            backend_manager.emit_progress(session_id, "Generating visualizations...", 90)
            dataset_name = dataset.replace('.csv', '')
            plots = pipeline.generate_plots(dataset_name, progress_callback)
            
            # Prepare results - convert to JSON-serializable format
            model_summary = pipeline.get_model_summary()
            
            # Convert numpy types in model_summary to Python types
            json_safe_summary = {}
            for model_name, metrics in model_summary.items():
                json_safe_summary[model_name] = {}
                for metric_name, value in metrics.items():
                    if isinstance(value, (np.integer, np.floating)):
                        json_safe_summary[model_name][metric_name] = float(value)
                    elif isinstance(value, np.ndarray):
                        json_safe_summary[model_name][metric_name] = value.tolist()
                    else:
                        json_safe_summary[model_name][metric_name] = value
            
            # Final progress update with results
            backend_manager.emit_progress(session_id, "Training completed!", 100, {
                'model_summary': json_safe_summary,
                'plots': plots,
                'dataset_info': pipeline.data_info
            })
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            with app.app_context():
                socketio.emit('training_error', {'error': str(e)}, room=session_id)
    
    # Start training in a separate thread
    training_thread = threading.Thread(target=training_worker)
    training_thread.daemon = True
    training_thread.start()

@app.route('/api/apply_cleaning', methods=['POST'])
def apply_cleaning():
    """Apply data cleaning operations"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        cleaning_operations = data.get('operations', {})
        
        file_path = os.path.join('data', filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        df = pd.read_csv(file_path)
        original_shape = df.shape
        
        # Apply cleaning operations in sequence
        results = {}
        
        # 1. Drop columns
        if 'drop_columns' in cleaning_operations:
            df, dropped = backend_manager.data_cleaner.drop_columns(
                df, cleaning_operations['drop_columns']
            )
            results['dropped_columns'] = dropped
        
        # 2. Rename columns
        if 'rename_columns' in cleaning_operations:
            df, renamed = backend_manager.data_cleaner.rename_columns(
                df, cleaning_operations['rename_columns']
            )
            results['renamed_columns'] = renamed
        
        # 3. Handle missing values
        if 'missing_values' in cleaning_operations:
            df, handled = backend_manager.data_cleaner.handle_missing_values(
                df, cleaning_operations['missing_values']
            )
            results['missing_values_handled'] = handled
        
        # 4. Encode categorical data
        if 'categorical_encoding' in cleaning_operations:
            df, encoded = backend_manager.data_cleaner.encode_categorical_data(
                df, cleaning_operations['categorical_encoding']
            )
            results['encoded_columns'] = encoded
        
        # 5. Apply transformations
        if 'transformations' in cleaning_operations:
            df, transformed = backend_manager.data_cleaner.apply_transformations(
                df, cleaning_operations['transformations']
            )
            results['transformed_columns'] = transformed
        
        # 6. Remove outliers if specified
        if 'remove_outliers' in cleaning_operations:
            outlier_indices = cleaning_operations['remove_outliers']
            df, removed_count = backend_manager.data_cleaner.remove_outliers(df, outlier_indices)
            results['outliers_removed'] = removed_count
        
        # Save cleaned dataset
        cleaned_filename = f"cleaned_{filename}"
        cleaned_path = os.path.join('data', cleaned_filename)
        df.to_csv(cleaned_path, index=False)
        
        # Validate data quality
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
    """Detect outliers in the dataset"""
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
    """Make predictions using trained models"""
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
        
        # Prepare feature array
        feature_array = np.array([[features.get(name, 0) for name in feature_names]])
        
        # Scale features if scaler exists
        if scaler:
            feature_array = scaler.transform(feature_array)
        
        # Make prediction
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
    """Export training results and model information"""
    try:
        # Generate export data
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
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(backend_manager.active_sessions)
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)