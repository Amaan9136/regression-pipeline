"""
Enhanced Flask Backend with Real-time WebSocket Support and Advanced Features
"""
from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
import os
import pickle
import json
import asyncio
import threading
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename
import logging

# Import our enhanced modules
from pipeline.regression_pipeline import AdvancedRegressionPipeline
from utils.data_cleaning import DataCleaner, DataValidator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
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

class BackendManager:
    """Manages backend operations and real-time communication"""
    
    def __init__(self):
        self.active_sessions = {}
        self.pipeline = None
        self.data_cleaner = DataCleaner()
        self.current_dataset = None
        self.cleaning_session = None
    
    def create_session(self, session_id):
        """Create a new session for a user"""
        self.active_sessions[session_id] = {
            'created_at': datetime.now(),
            'pipeline': AdvancedRegressionPipeline(),
            'data_cleaner': DataCleaner(),
            'current_step': 'idle',
            'progress': 0
        }
        return session_id
    
    def get_session(self, session_id):
        """Get session data"""
        return self.active_sessions.get(session_id)
    
    def emit_progress(self, session_id, message, progress, data=None):
        """Emit progress update to specific session"""
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
    return render_template('enhanced_index.html', datasets=datasets)

@app.route('/api/preview_data', methods=['POST'])
def preview_data():
    """Preview uploaded dataset"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are allowed'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join('temp', f"preview_{filename}")
        file.save(temp_path)
        
        # Preview data
        preview_info = backend_manager.data_cleaner.preview_data(temp_path)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return jsonify(preview_info)
        
    except Exception as e:
        logger.error(f"Error previewing data: {str(e)}")
        return jsonify({'error': f'Preview failed: {str(e)}'}), 500

@app.route('/api/upload_dataset', methods=['POST'])
def upload_dataset():
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
        df = pd.read_csv(file_path, nrows=100)  # Quick preview
        file_info = {
            'filename': filename,
            'size': os.path.getsize(file_path),
            'columns': list(df.columns),
            'shape_preview': df.shape,
            'upload_time': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'file_info': file_info
        })
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/get_column_info/<filename>')
def get_column_info(filename):
    """Get detailed column information for data cleaning interface"""
    try:
        file_path = os.path.join('data', filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        df = pd.read_csv(file_path)
        column_info = backend_manager.data_cleaner.get_column_info(df)
        
        return jsonify({
            'success': True,
            'column_info': column_info,
            'dataset_shape': df.shape
        })
        
    except Exception as e:
        logger.error(f"Error getting column info: {str(e)}")
        return jsonify({'error': f'Failed to get column info: {str(e)}'}), 500

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
        method = request.args.get('method', 'iqr')
        threshold = float(request.args.get('threshold', 1.5))
        
        outliers_info = backend_manager.data_cleaner.detect_outliers(df, method, threshold)
        
        return jsonify({
            'success': True,
            'outliers': outliers_info
        })
        
    except Exception as e:
        logger.error(f"Error detecting outliers: {str(e)}")
        return jsonify({'error': f'Outlier detection failed: {str(e)}'}), 500

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
    
    session = backend_manager.get_session(session_id)
    if not session:
        emit('training_error', {'error': 'Invalid session'})
        return
    
    def training_worker():
        try:
            dataset_path = os.path.join('data', dataset)
            if not os.path.exists(dataset_path):
                emit('training_error', {'error': 'Dataset not found'})
                return
            
            # Initialize pipeline with custom config
            pipeline = AdvancedRegressionPipeline(config)
            
            # Progress callback for real-time updates
            def progress_callback(message, progress):
                backend_manager.emit_progress(session_id, message, progress)
            
            # Load and preprocess data
            backend_manager.emit_progress(session_id, "Loading dataset...", 5)
            X, y, df = pipeline.load_and_preprocess_data(dataset_path)
            
            if X is None:
                emit('training_error', {'error': 'Failed to load dataset'})
                return
            
            backend_manager.emit_progress(session_id, "Dataset loaded successfully", 10)
            
            # Train models
            backend_manager.emit_progress(session_id, "Starting model training...", 15)
            pipeline.train_models(X, y, progress_callback)
            
            # Generate plots
            backend_manager.emit_progress(session_id, "Generating visualizations...", 90)
            dataset_name = dataset.replace('.csv', '')
            plots = pipeline.generate_advanced_plots(dataset_name, progress_callback)
            
            # Prepare results
            model_summary = pipeline.get_model_summary()
            
            # Final progress update with results
            backend_manager.emit_progress(session_id, "Training completed!", 100, {
                'model_summary': model_summary,
                'plots': plots,
                'dataset_info': {
                    'filename': dataset,
                    'shape': df.shape,
                    'features': list(X.columns),
                    'target': df.columns[-1] if 'target_name' not in pipeline.data_info else pipeline.data_info['target_name']
                }
            })
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            emit('training_error', {'error': f'Training failed: {str(e)}'})
    
    # Start training in a separate thread
    training_thread = threading.Thread(target=training_worker)
    training_thread.daemon = True
    training_thread.start()

@app.route('/api/model_comparison/<dataset_name>')
def model_comparison(dataset_name):
    """Get detailed model comparison data"""
    try:
        # This would typically load from saved results
        # For now, return mock data structure
        return jsonify({
            'success': True,
            'comparison_data': {
                'models': [],
                'metrics': [],
                'best_model': '',
                'cross_validation_scores': {}
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_details/<model_name>')
def model_details(model_name):
    """Get detailed information about a specific model"""
    try:
        model_path = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
        
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found'}), 404
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        return jsonify({
            'success': True,
            'model_details': {
                'name': model_name,
                'parameters': model_data.get('best_params', {}),
                'training_time': model_data.get('timestamp', ''),
                'config': model_data.get('config', {}),
                'data_info': model_data.get('data_info', {})
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting model details: {str(e)}")
        return jsonify({'error': f'Failed to get model details: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions using a trained model"""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        features = data.get('features')  # Dictionary of feature values
        
        model_path = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
        
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found'}), 404
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Prepare features for prediction
        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        # Scale features
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
        # This would generate a comprehensive report
        # For now, return success message
        return jsonify({
            'success': True,
            'message': 'Results exported successfully',
            'export_path': f'exports/{dataset_name}_results.json'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
    # Run with SocketIO support
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)