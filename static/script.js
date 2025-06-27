class MLPipelineApp {
    constructor() {
        this.socket = io();
        this.sessionId = null;
        this.currentDataset = null;
        this.trainingInProgress = false;
        this.currentStep = 'upload';
        this.datasetInfo = null;
        this.cleaningOperations = {};
        this.modelResults = {};
        this.visualizations = {};
        this.predictionModels = [];
        this.activeTab = 'upload';
        this.healthCheckInterval = null;
        this.connectionEstablished = false;
        this.isUploading = false;
        
        this.config = {
            ALERT_AUTO_REMOVE_DELAY: 5000,
            MAX_ACTIVITY_LOGS: 50,
            HEALTH_CHECK_INTERVAL: 30000,
            PROGRESS_UPDATE_INTERVAL: 100
        };
        
        this.initialize();
    }
    
    initialize() {
        this.setupEventListeners();
        this.setupWebSocketHandlers();
        this.initializeUI();
        this.startHealthCheck();
        console.log('🚀 ML Pipeline Application Initialized');
    }
    
    setupEventListeners() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.bindEventListeners());
        } else {
            this.bindEventListeners();
        }
    }
    
    bindEventListeners() {
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const section = e.target.dataset.section;
                if (section) this.switchSection(section);
            });
        });
        
        this.setupFileUpload();
        this.setupDataPreprocessing();
        this.setupTrainingHandlers();
        this.setupResultsHandlers();
        this.setupPredictionHandlers();
        this.setupSidePanelHandlers();
        this.setupModalHandlers();
        
        console.log('✅ Event listeners bound successfully');
    }
    
    setupWebSocketHandlers() {
        this.socket.on('connect', () => {
            console.log('🔗 Connected to server');
            this.connectionEstablished = true;
            this.updateConnectionStatus(true);
            this.addActivityLog('Connected to server', 'success');
        });
        
        this.socket.on('disconnect', () => {
            console.log('❌ Disconnected from server');
            this.connectionEstablished = false;
            this.updateConnectionStatus(false);
            this.showAlert('Connection lost. Attempting to reconnect...', 'warning');
        });
        
        this.socket.on('session_created', (data) => {
            this.sessionId = data.session_id;
            this.updateSessionInfo();
            this.addActivityLog('Session created successfully', 'success');
            console.log(`📱 Session created: ${this.sessionId}`);
        });
        
        this.socket.on('training_progress', (data) => {
            this.updateTrainingProgress(data);
            this.addActivityLog(data.message, 'info', data.progress);
            
            if (data.progress === 100 && data.data) {
                this.handleTrainingCompletion(data.data);
            }
        });
        
        this.socket.on('training_error', (data) => {
            this.handleTrainingError(data.error);
        });
        
        this.socket.on('data_update', (data) => {
            this.handleDataUpdate(data);
        });
        
        this.socket.on('system_status', (data) => {
            this.updateSystemStatus(data);
        });
    }
    
    setupFileUpload() {
        const fileInput = document.getElementById('file-input');
        const uploadZone = document.getElementById('upload-zone');
        const datasetSelect = document.getElementById('dataset-select');
        
        if (uploadZone) {
            uploadZone.addEventListener('dragover', this.handleDragOver.bind(this));
            uploadZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
            uploadZone.addEventListener('drop', this.handleFileDrop.bind(this));
            uploadZone.addEventListener('click', () => {
                if (!this.isUploading && fileInput) {
                    fileInput.click();
                }
            });
        }
        
        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                if (!this.isUploading) {
                    this.processFiles(Array.from(e.target.files));
                }
            });
        }
        
        if (datasetSelect) {
            datasetSelect.addEventListener('change', (e) => {
                this.selectDataset(e.target.value);
            });
        }
    }
    
    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        if (!this.isUploading) {
            e.currentTarget.classList.add('dragover');
        }
    }
    
    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.remove('dragover');
    }
    
    handleFileDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.remove('dragover');
        if (!this.isUploading) {
            const files = Array.from(e.dataTransfer.files);
            this.processFiles(files);
        }
    }
    
    async processFiles(files) {
        if (this.isUploading) return;
        
        const csvFiles = files.filter(file => file.name.endsWith('.csv'));
        
        if (csvFiles.length === 0) {
            this.showAlert('Please select CSV files only.', 'warning');
            return;
        }
        
        for (const file of csvFiles) {
            await this.uploadFile(file);
        }
    }
    
    async uploadFile(file) {
        if (this.isUploading) return;
        
        try {
            this.isUploading = true;
            this.showLoading(`Uploading ${file.name}...`);
            
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showAlert(`File uploaded successfully: ${result.filename}`, 'success');
                this.currentDataset = result.filename;
                this.datasetInfo = result.preview;
                this.displayDatasetPreview(result.preview);
                this.updateDatasetStats();
                this.addActivityLog(`Dataset uploaded: ${result.filename}`, 'success');
                this.refreshDatasetSelect();
            } else {
                throw new Error(result.error || 'Upload failed');
            }
        } catch (error) {
            this.showAlert(`Upload failed: ${error.message}`, 'error');
            console.error('Upload error:', error);
        } finally {
            this.isUploading = false;
            this.hideLoading();
        }
    }
    
    async selectDataset(filename) {
        if (!filename) return;
        
        try {
            this.showLoading('Loading dataset information...');
            
            const response = await fetch(`/api/dataset_info/${filename}`);
            const result = await response.json();
            
            if (result.success) {
                this.currentDataset = filename;
                this.datasetInfo = result.info;
                
                // Debug logging
                console.log('Dataset info received:', result.info);
                console.log('Columns type:', typeof result.info.columns, result.info.columns);
                
                this.displayDatasetPreview(result.info);
                this.updateDatasetStats();
                this.addActivityLog(`Dataset selected: ${filename}`, 'info');
            } else {
                throw new Error(result.error || 'Failed to load dataset info');
            }
        } catch (error) {
            this.showAlert(`Error loading dataset: ${error.message}`, 'error');
            console.error('Dataset selection error:', error);
        } finally {
            this.hideLoading();
        }
    }
    
    refreshDatasetSelect() {
        fetch('/health').then(() => {
            location.reload();
        }).catch(console.error);
    }
    
    setupDataPreprocessing() {
        const detectOutliersBtn = document.getElementById('detect-outliers');
        const applyCleaningBtn = document.getElementById('apply-cleaning');
        
        if (detectOutliersBtn) {
            detectOutliersBtn.addEventListener('click', () => this.detectOutliers());
        }
        
        if (applyCleaningBtn) {
            applyCleaningBtn.addEventListener('click', () => this.applyDataCleaning());
        }
        
        document.querySelectorAll('.cleaning-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchCleaningTab(e.target.dataset.tab);
            });
        });
    }
    
    async detectOutliers() {
        if (!this.currentDataset) {
            this.showAlert('Please select a dataset first', 'warning');
            return;
        }
        
        try {
            this.showLoading('Detecting outliers...');
            
            const response = await fetch(`/api/detect_outliers/${this.currentDataset}`);
            const result = await response.json();
            
            if (result.success) {
                this.displayOutliers(result.outliers);
                this.addActivityLog('Outliers detected successfully', 'success');
            } else {
                throw new Error(result.error || 'Outlier detection failed');
            }
        } catch (error) {
            this.showAlert(`Outlier detection failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    async applyDataCleaning() {
        if (!this.currentDataset) {
            this.showAlert('Please select a dataset first', 'warning');
            return;
        }
        
        try {
            this.showLoading('Applying data cleaning operations...');
            
            const response = await fetch('/api/apply_cleaning', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    filename: this.currentDataset,
                    operations: this.cleaningOperations
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.displayCleaningResults(result);
                this.addActivityLog('Data cleaning applied successfully', 'success');
                this.currentDataset = result.cleaned_filename;
            } else {
                throw new Error(result.error || 'Data cleaning failed');
            }
        } catch (error) {
            this.showAlert(`Data cleaning failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    setupTrainingHandlers() {
        const startTrainingBtn = document.getElementById('start-training');
        const stopTrainingBtn = document.getElementById('stop-training');
        
        if (startTrainingBtn) {
            startTrainingBtn.addEventListener('click', () => this.startTraining());
        }
        
        if (stopTrainingBtn) {
            stopTrainingBtn.addEventListener('click', () => this.stopTraining());
        }
        
        this.setupTrainingConfiguration();
    }
    
    setupTrainingConfiguration() {
        const configForm = document.getElementById('training-config-form');
        if (configForm) {
            configForm.addEventListener('change', () => this.updateTrainingConfig());
        }
        
        const advancedToggle = document.getElementById('advanced-config-toggle');
        if (advancedToggle) {
            advancedToggle.addEventListener('change', (e) => {
                this.toggleAdvancedConfig(e.target.checked);
            });
        }
    }
    
    updateTrainingConfig() {
        const config = {
            test_size: parseFloat(document.getElementById('test-size')?.value || 0.2),
            random_state: parseInt(document.getElementById('random-state')?.value || 42),
            cross_validation_folds: parseInt(document.getElementById('cv-folds')?.value || 5),
            hyperparameter_tuning: document.getElementById('hyperparameter-tuning')?.checked || true,
            feature_engineering: document.getElementById('feature-engineering')?.checked || true,
            scaling_method: document.getElementById('scaling-method')?.value || 'standard',
            feature_selection_k: document.getElementById('feature-selection-k')?.value || 'all'
        };
        
        this.trainingConfig = config;
        console.log('Training configuration updated:', config);
    }
    
    async startTraining() {
        if (!this.currentDataset) {
            this.showAlert('Please select a dataset first', 'warning');
            return;
        }
        
        if (!this.sessionId) {
            this.showAlert('Session not established. Please refresh the page.', 'error');
            return;
        }
        
        if (this.trainingInProgress) {
            this.showAlert('Training already in progress', 'warning');
            return;
        }
        
        try {
            this.trainingInProgress = true;
            this.updateTrainingUI(true);
            
            const config = this.trainingConfig || {};
            
            this.socket.emit('start_training', {
                session_id: this.sessionId,
                dataset: this.currentDataset,
                config: config
            });
            
            this.addActivityLog('Training started', 'info');
            this.showAlert('Training started successfully', 'success');
            
        } catch (error) {
            this.showAlert(`Failed to start training: ${error.message}`, 'error');
            this.trainingInProgress = false;
            this.updateTrainingUI(false);
        }
    }
    
    stopTraining() {
        this.trainingInProgress = false;
        this.updateTrainingUI(false);
        this.addActivityLog('Training stopped', 'warning');
        this.showAlert('Training stopped', 'warning');
    }
    
    updateTrainingUI(inProgress) {
        const startBtn = document.getElementById('start-training');
        const stopBtn = document.getElementById('stop-training');
        const progressContainer = document.getElementById('training-progress-container');
        
        if (startBtn) startBtn.disabled = inProgress;
        if (stopBtn) stopBtn.disabled = !inProgress;
        if (progressContainer) {
            progressContainer.style.display = inProgress ? 'block' : 'none';
        }
    }
    
    handleTrainingCompletion(data) {
        this.trainingInProgress = false;
        this.updateTrainingUI(false);
        
        // Store results with fallback handling
        this.modelResults = data.model_summary || data.model_summaries || {};
        this.visualizations = data.plots || {};
        
        console.log('Training completed with data:', data);
        
        try {
            this.displayTrainingResults(data);
            this.addActivityLog('Training completed successfully!', 'success');
            this.showAlert('Training completed successfully!', 'success');
            this.switchSection('results');
        } catch (error) {
            console.error('Error displaying training results:', error);
            this.showAlert('Training completed, but there was an error displaying results', 'warning');
        }
    }
    
    handleTrainingError(error) {
        this.trainingInProgress = false;
        this.updateTrainingUI(false);
        this.showAlert(`Training failed: ${error}`, 'error');
        this.addActivityLog(`Training error: ${error}`, 'error');
        console.error('Training error:', error);
    }
    
    setupResultsHandlers() {
        const exportBtn = document.getElementById('export-results');
        const downloadBtn = document.getElementById('download-model');
        
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportResults());
        }
        
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => this.downloadModel());
        }
    }
    
    setupPredictionHandlers() {
        const makePredictionBtn = document.getElementById('make-prediction');
        const batchPredictionBtn = document.getElementById('batch-prediction');
        
        if (makePredictionBtn) {
            makePredictionBtn.addEventListener('click', () => this.makePrediction());
        }
        
        if (batchPredictionBtn) {
            batchPredictionBtn.addEventListener('click', () => this.makeBatchPrediction());
        }
    }
    
    async makePrediction() {
        const modelSelect = document.getElementById('prediction-model-select');
        const featureInputs = document.querySelectorAll('.feature-input');
        
        if (!modelSelect || !modelSelect.value) {
            this.showAlert('Please select a model for prediction', 'warning');
            return;
        }
        
        const features = {};
        featureInputs.forEach(input => {
            features[input.name] = parseFloat(input.value) || 0;
        });
        
        try {
            this.showLoading('Making prediction...');
            
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_name: modelSelect.value,
                    features: features
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.displayPredictionResult(result);
                this.addActivityLog(`Prediction made: ${result.prediction}`, 'success');
            } else {
                throw new Error(result.error || 'Prediction failed');
            }
        } catch (error) {
            this.showAlert(`Prediction failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    async exportResults() {
        if (!this.currentDataset) {
            this.showAlert('No results to export', 'warning');
            return;
        }
        
        try {
            const response = await fetch(`/api/export_results/${this.currentDataset}`);
            const result = await response.json();
            
            if (result.success) {
                this.showAlert('Results exported successfully', 'success');
                this.addActivityLog('Results exported', 'success');
            } else {
                throw new Error(result.error || 'Export failed');
            }
        } catch (error) {
            this.showAlert(`Export failed: ${error.message}`, 'error');
        }
    }
    
    setupSidePanelHandlers() {
        const panelToggle = document.getElementById('panel-toggle');
        const sidePanel = document.getElementById('side-panel');
        
        if (panelToggle && sidePanel) {
            panelToggle.addEventListener('click', () => {
                sidePanel.classList.toggle('collapsed');
                panelToggle.textContent = sidePanel.classList.contains('collapsed') ? '▶' : '◀';
            });
        }
    }
    
    setupModalHandlers() {
        document.querySelectorAll('.modal-close').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const modal = e.target.closest('.modal');
                if (modal) this.closeModal(modal.id);
            });
        });
        
        document.querySelectorAll('.modal').forEach(modal => {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.closeModal(modal.id);
                }
            });
        });
    }
    
    showModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.style.display = 'flex';
            document.body.style.overflow = 'hidden';
        }
    }
    
    closeModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.style.display = 'none';
            document.body.style.overflow = 'auto';
        }
    }
    
    initializeUI() {
        this.updateConnectionStatus(false);
        this.updateTrainingUI(false);
        
        const uploadSection = document.getElementById('upload-section');
        if (uploadSection) {
            uploadSection.classList.add('active');
        }
        
        const firstTab = document.querySelector('.nav-tab');
        if (firstTab) {
            firstTab.classList.add('active');
        }
    }
    
    switchSection(sectionName) {
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-section="${sectionName}"]`)?.classList.add('active');
        
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });
        document.getElementById(`${sectionName}-section`)?.classList.add('active');
        
        this.activeTab = sectionName;
        this.addActivityLog(`Switched to ${sectionName} section`, 'info');
    }
    
    switchCleaningTab(tabName) {
        document.querySelectorAll('.cleaning-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`)?.classList.add('active');
        
        document.querySelectorAll('.cleaning-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-content`)?.classList.add('active');
    }
    
    toggleAdvancedConfig(show) {
        const advancedPanel = document.getElementById('advanced-config-panel');
        if (advancedPanel) {
            advancedPanel.style.display = show ? 'block' : 'none';
        }
    }
    
    showAlert(message, type = 'info') {
        const notifications = document.getElementById('notifications');
        if (!notifications) return;
        
        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.innerHTML = `
            <span class="alert-message">${message}</span>
            <button class="alert-close">&times;</button>
        `;
        
        notifications.appendChild(alert);
        
        const closeBtn = alert.querySelector('.alert-close');
        closeBtn.addEventListener('click', () => {
            alert.remove();
        });
        
        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, this.config.ALERT_AUTO_REMOVE_DELAY);
    }
    
    showLoading(title = 'Processing...', message = 'Please wait while we process your request.') {
        const overlay = document.getElementById('loading-overlay');
        const titleEl = document.getElementById('loading-title');
        const messageEl = document.getElementById('loading-message');
        
        if (overlay) overlay.style.display = 'flex';
        if (titleEl) titleEl.textContent = title;
        if (messageEl) messageEl.textContent = message;
    }
    
    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) overlay.style.display = 'none';
    }
    
    updateConnectionStatus(connected) {
        const statusIndicator = document.getElementById('health-indicator');
        const connectionStatus = document.getElementById('connection-status');
        
        if (statusIndicator) {
            const statusDot = statusIndicator.querySelector('.status-dot');
            const statusText = statusIndicator.querySelector('.status-text');
            
            if (statusDot) {
                statusDot.className = `status-dot ${connected ? 'status-connected' : 'status-disconnected'}`;
            }
            
            if (statusText) {
                statusText.textContent = connected ? 'Connected' : 'Disconnected';
            }
        }
        
        if (connectionStatus) {
            connectionStatus.textContent = connected ? 'Yes' : 'No';
        }
    }
    
    updateSessionInfo() {
        const sessionIdSpan = document.getElementById('session-id');
        if (sessionIdSpan && this.sessionId) {
            sessionIdSpan.textContent = this.sessionId.substring(0, 8) + '...';
        }
    }
    
    updateDatasetStats() {
        if (!this.datasetInfo) return;
        
        const datasetsProcessed = document.getElementById('datasets-processed');
        if (datasetsProcessed) {
            datasetsProcessed.textContent = '1';
        }
    }
    
    displayDatasetPreview(preview) {
        const previewContainer = document.getElementById('dataset-preview');
        if (!previewContainer || !preview) return;
        
        // Ensure columns is always an array
        let columns = preview.columns;
        if (!Array.isArray(columns)) {
            if (preview.sample_data && preview.sample_data.length > 0) {
                columns = Object.keys(preview.sample_data[0]);
            } else {
                columns = [];
            }
        }
        
        let html = `
            <div class="dataset-info">
                <h4>📊 Dataset Preview</h4>
                <div class="dataset-stats">
                    <span class="stat-item">Rows: ${preview.shape ? preview.shape[0] : 'N/A'}</span>
                    <span class="stat-item">Columns: ${preview.shape ? preview.shape[1] : columns.length}</span>
                </div>
            </div>
        `;
        
        if (preview.sample_data && preview.sample_data.length > 0 && columns.length > 0) {
            html += '<div class="preview-table-container">';
            html += '<table class="preview-table">';
            html += '<thead><tr>';
            
            columns.forEach(col => {
                html += `<th>${col}</th>`;
            });
            
            html += '</tr></thead><tbody>';
            
            preview.sample_data.slice(0, 5).forEach(row => {
                html += '<tr>';
                columns.forEach(col => {
                    const value = row[col];
                    html += `<td>${value !== undefined && value !== null ? value : '-'}</td>`;
                });
                html += '</tr>';
            });
            
            html += '</tbody></table></div>';
        } else {
            html += '<p class="text-muted">No preview data available</p>';
        }
        
        previewContainer.innerHTML = html;
        previewContainer.classList.add('show');
    }
    
    displayOutliers(outliers) {
        const outliersContainer = document.getElementById('outliers-display');
        if (!outliersContainer) return;
        
        let html = '<div class="outliers-summary"><h4>🔍 Detected Outliers</h4>';
        
        if (!outliers || Object.keys(outliers).length === 0) {
            html += '<p>No outliers detected in the dataset.</p>';
        } else {
            html += '<div class="outliers-list">';
            Object.entries(outliers).forEach(([column, data]) => {
                const count = data.count || 0;
                const indices = data.indices || [];
                const percentage = data.percentage || 0;
                
                // Ensure indices is an array
                const indicesArray = Array.isArray(indices) ? indices : [];
                
                html += `
                    <div class="outlier-column">
                        <h5>${column}</h5>
                        <p>${count} outliers found (${percentage.toFixed(2)}% of data)</p>
                        ${indicesArray.length > 0 ? `
                            <p>Sample indices: ${indicesArray.slice(0, 10).join(', ')}${indicesArray.length > 10 ? '...' : ''}</p>
                            <div class="outlier-actions">
                                <button class="btn btn-sm btn-secondary" onclick="app.viewColumnOutliers('${column}')">
                                    View Details
                                </button>
                                <button class="btn btn-sm btn-danger" onclick="app.removeColumnOutliers('${column}', ${JSON.stringify(indicesArray)})">
                                    Remove Outliers
                                </button>
                            </div>
                        ` : ''}
                    </div>
                `;
            });
            html += '</div>';
        }
        
        html += '</div>';
        outliersContainer.innerHTML = html;
    }
    
    viewColumnOutliers(column) {
        this.showAlert(`Viewing outliers for column: ${column}`, 'info');
    }
    
    removeColumnOutliers(column, indices) {
        if (!this.cleaningOperations.remove_outliers) {
            this.cleaningOperations.remove_outliers = [];
        }
        
        // Ensure indices is an array
        const indicesArray = Array.isArray(indices) ? indices : [];
        this.cleaningOperations.remove_outliers = [...new Set([...this.cleaningOperations.remove_outliers, ...indicesArray])];
        
        this.showAlert(`Marked ${indicesArray.length} outliers from ${column} for removal`, 'success');
        this.addActivityLog(`Marked outliers for removal: ${column}`, 'info');
    }
    
    displayCleaningResults(results) {
        const resultsContainer = document.getElementById('cleaning-results');
        if (!resultsContainer) return;
        
        let html = `
            <div class="cleaning-results-summary">
                <h4>🧹 Data Cleaning Results</h4>
                <div class="shape-comparison">
                    <span>Original: ${results.original_shape[0]} rows × ${results.original_shape[1]} columns</span>
                    <span>→</span>
                    <span>Final: ${results.final_shape[0]} rows × ${results.final_shape[1]} columns</span>
                </div>
                <div class="cleaning-operations">
                    ${Object.entries(results.results).map(([operation, result]) => `
                        <div class="operation-result">
                            <strong>${operation}:</strong> ${JSON.stringify(result)}
                        </div>
                    `).join('')}
                </div>
                ${results.cleaning_summary ? this.renderCleaningSummary(results.cleaning_summary) : ''}
            </div>
        `;
        
        resultsContainer.innerHTML = html;
    }
    
    displayTrainingResults(data) {
        const resultsContainer = document.getElementById('training-results');
        if (!resultsContainer) return;
        
        console.log('Displaying training results:', data);
        
        let html = '<div class="training-results-summary"><h4>🎯 Training Results</h4>';
        
        // Handle the training summary
        const trainingSummary = data.training_summary || {};
        const modelSummary = data.model_summary || trainingSummary.model_summary || {};
        
        // Display overall summary
        html += '<div class="training-overview">';
        if (trainingSummary.training_completed) {
            html += `<div class="success-message">✅ Training completed successfully!</div>`;
            html += `<div class="summary-stats">`;
            html += `<span class="stat">Models Trained: <strong>${trainingSummary.models_trained || Object.keys(modelSummary).length}</strong></span>`;
            if (trainingSummary.best_model_name && trainingSummary.best_model_name !== 'None') {
                html += `<span class="stat">Best Model: <strong>${trainingSummary.best_model_name}</strong></span>`;
                html += `<span class="stat">Best Score: <strong>${(trainingSummary.best_score || 0).toFixed(4)}</strong></span>`;
            }
            html += `</div>`;
        } else {
            html += `<div class="error-message">❌ Training failed</div>`;
            if (trainingSummary.error) {
                html += `<div class="error-details">${trainingSummary.error}</div>`;
            }
        }
        html += '</div>';

        // Display model comparison table
        if (Object.keys(modelSummary).length > 0) {
            html += '<div class="model-comparison-table">';
            html += '<h5>📊 Model Performance Comparison</h5>';
            html += '<table class="comparison-table">';
            html += '<thead><tr><th>Model</th><th>R² Score</th><th>RMSE</th><th>MAE</th><th>CV Mean</th><th>CV Std</th></tr></thead>';
            html += '<tbody>';
            
            // Sort models by R² score
            const sortedModels = Object.entries(modelSummary).sort((a, b) => (b[1].r2_score || 0) - (a[1].r2_score || 0));
            
            sortedModels.forEach(([modelName, metrics], index) => {
                const isBest = index === 0;
                html += `<tr class="${isBest ? 'best-model' : ''}">`;
                html += `<td>${modelName}${isBest ? ' 🏆' : ''}</td>`;
                html += `<td>${(metrics.r2_score || 0).toFixed(4)}</td>`;
                html += `<td>${(metrics.rmse || 0).toFixed(4)}</td>`;
                html += `<td>${(metrics.mae || 0).toFixed(4)}</td>`;
                html += `<td>${(metrics.cv_mean || 0).toFixed(4)}</td>`;
                html += `<td>${(metrics.cv_std || 0).toFixed(4)}</td>`;
                html += '</tr>';
            });
            
            html += '</tbody></table>';
            html += '</div>';
            
            // Update prediction models for the prediction section
            this.predictionModels = Object.keys(modelSummary);
            this.updatePredictionModelSelect();
        }

        // Display dataset info
        if (data.dataset_info) {
            html += '<div class="dataset-info">';
            html += '<h5>📋 Dataset Information</h5>';
            html += `<div class="info-grid">`;
            html += `<span>Shape: <strong>${data.dataset_info.shape[0]} rows × ${data.dataset_info.shape[1]} columns</strong></span>`;
            html += `<span>Target Column: <strong>${data.dataset_info.target_column}</strong></span>`;
            html += `<span>Features: <strong>${data.dataset_info.features.length}</strong></span>`;
            html += `</div>`;
            html += '</div>';
        }

        html += '</div>';
        resultsContainer.innerHTML = html;

        // Display visualizations
        this.displayVisualizations(data.plots || {});
        
        // Store results for later use
        this.modelResults = modelSummary;
        this.datasetInfo = data.dataset_info;
    }

    displayVisualizations(plots) {
        const vizContainer = document.getElementById('visualizations-container');
        if (!vizContainer) return;

        let html = '<div class="visualizations"><h3>📈 Visualizations</h3>';

        if (Object.keys(plots).length > 0) {
            html += '<div class="viz-grid">';
            
            Object.entries(plots).forEach(([plotType, plotData]) => {
                html += `<div class="viz-card">`;
                html += `<h4>${this.formatPlotTitle(plotType)}</h4>`;
                html += `<div class="viz-content">`;
                
                if (plotData && plotData !== 'No data available') {
                    if (plotData.startsWith('static/') || plotData.startsWith('/static/')) {
                        // Image path
                        html += `<img src="${plotData}" alt="${plotType}" style="max-width: 100%; height: auto;">`;
                    } else {
                        // Plot data or placeholder
                        html += `<div class="viz-placeholder">📊 ${plotData}</div>`;
                    }
                } else {
                    html += `<div class="viz-placeholder">No ${plotType} data available</div>`;
                }
                
                html += `</div></div>`;
            });
            
            html += '</div>';
        } else {
            html += '<div class="no-visualizations">No visualizations available</div>';
        }

        html += '</div>';
        vizContainer.innerHTML = html;
    }

    updatePredictionModelSelect() {
        const modelSelect = document.getElementById('prediction-model-select');
        if (!modelSelect) return;

        // Clear existing options except the first one
        modelSelect.innerHTML = '<option value="">Choose a trained model...</option>';

        // Add model options
        this.predictionModels.forEach(modelName => {
            const option = document.createElement('option');
            option.value = modelName;
            option.textContent = modelName;
            modelSelect.appendChild(option);
        });

        // Update feature inputs based on dataset info
        this.updateFeatureInputs();
    }

    updateFeatureInputs() {
        const featureInputsContainer = document.getElementById('feature-inputs');
        if (!featureInputsContainer || !this.datasetInfo) return;

        let html = '<h4>Feature Values</h4>';
        html += '<div class="feature-input-grid">';

        this.datasetInfo.features.forEach(feature => {
            html += `<div class="input-group">`;
            html += `<label for="feature-${feature}">${feature}</label>`;
            html += `<input type="number" id="feature-${feature}" name="${feature}" 
                    class="feature-input" step="any" placeholder="Enter ${feature}">`;
            html += `</div>`;
        });

        html += '</div>';
        featureInputsContainer.innerHTML = html;
    }

    displayPredictionResult(result) {
        const resultContainer = document.getElementById('prediction-result');
        if (!resultContainer) return;

        const html = `
            <div class="prediction-result">
                <h4>🔮 Prediction Result</h4>
                <div class="result-value">
                    <strong>${result.prediction.toFixed(4)}</strong>
                </div>
                <div class="result-meta">
                    <span>Model: ${result.model_name}</span>
                    <span>Confidence: ${(result.confidence || 0.95).toFixed(3)}</span>
                </div>
            </div>
        `;
        
        resultContainer.innerHTML = html;
    }

    formatPlotTitle(plotType) {
        const titles = {
            'model_comparison': 'Model Performance Comparison',
            'feature_importance': 'Feature Importance Analysis',
            'residual_plots': 'Residual Analysis',
            'best_scatter': 'Predicted vs Actual (Best Model)',
            'heatmap': 'Performance Metrics Heatmap'
        };
        return titles[plotType] || plotType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    updateTrainingProgress(data) {
        const progressBar = document.getElementById('training-progress-bar');
        const progressText = document.getElementById('training-progress-text');
        const progressPercentage = document.getElementById('training-progress-percentage');
        
        if (progressBar) {
            progressBar.style.width = `${data.progress}%`;
            progressBar.style.transition = 'width 0.3s ease-in-out';
        }
        
        if (progressText) {
            progressText.textContent = data.message;
            // Add fade effect for new messages
            progressText.style.opacity = '0.7';
            setTimeout(() => {
                if (progressText) progressText.style.opacity = '1';
            }, 100);
        }
        
        if (progressPercentage) {
            progressPercentage.textContent = `${Math.round(data.progress)}%`;
        }
        
        // Add live activity log
        this.addActivityLog(data.message, 'info', data.progress);
        
        console.log(`Training progress: ${data.progress}% - ${data.message}`);
        
        // Auto-scroll to show progress if not visible
        const progressContainer = document.getElementById('training-progress-container');
        if (progressContainer && data.progress > 0) {
            progressContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    addActivityLog(message, type = 'info', progress = null) {
        const activityList = document.getElementById('activity-list');
        if (!activityList) return;

        const timestamp = new Date().toLocaleTimeString();
        const progressInfo = progress !== null ? ` (${Math.round(progress)}%)` : '';
        
        const logItem = document.createElement('div');
        logItem.className = `activity-item ${type}`;
        logItem.innerHTML = `
            <span class="activity-time">${timestamp}</span>
            <span class="activity-message">${message}${progressInfo}</span>
            <span class="activity-type">${this.getActivityIcon(type)}</span>
        `;
        
        // Insert at top
        activityList.insertBefore(logItem, activityList.firstChild);
        
        // Limit logs
        const logs = activityList.querySelectorAll('.activity-item');
        if (logs.length > this.config.MAX_ACTIVITY_LOGS) {
            logs[logs.length - 1].remove();
        }
        
        // Auto-remove old logs
        setTimeout(() => {
            if (logItem.parentNode) {
                logItem.style.opacity = '0.5';
            }
        }, this.config.ALERT_AUTO_REMOVE_DELAY * 2);
    }

    getActivityIcon(type) {
        const icons = {
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️'
        };
        return icons[type] || 'ℹ️';
    }

    populateModelSelectors(modelNames) {
        const modelSelect = document.getElementById('prediction-model-select');
        if (modelSelect && modelNames.length > 0) {
            modelSelect.innerHTML = '<option value="">Choose a trained model...</option>';
            modelNames.forEach(modelName => {
                const displayName = modelName.replace(/_/g, ' ');
                modelSelect.innerHTML += `<option value="${modelName}">${displayName}</option>`;
            });
        }
    }

    formatMetric(value) {
        if (typeof value === 'number') {
            return value.toFixed(4);
        }
        return value || 'N/A';
    }

    formatMetricName(name) {
        return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
    
    async startHealthCheck() {
        this.healthCheckInterval = setInterval(async () => {
            try {
                const response = await fetch('/health');
                const result = await response.json();
                
                if (result.status === 'healthy') {
                    const healthIndicator = document.getElementById('health-indicator');
                    if (healthIndicator) {
                        const statusDot = healthIndicator.querySelector('.status-dot');
                        const statusText = healthIndicator.querySelector('.status-text');
                        
                        if (statusDot) statusDot.className = 'status-dot status-healthy';
                        if (statusText) statusText.textContent = 'Healthy';
                    }
                    
                    const activeSessions = document.getElementById('active-sessions');
                    if (activeSessions) {
                        activeSessions.textContent = result.active_sessions || 0;
                    }
                }
            } catch (error) {
                console.warn('Health check failed:', error);
                const healthIndicator = document.getElementById('health-indicator');
                if (healthIndicator) {
                    const statusDot = healthIndicator.querySelector('.status-dot');
                    const statusText = healthIndicator.querySelector('.status-text');
                    
                    if (statusDot) statusDot.className = 'status-dot status-error';
                    if (statusText) statusText.textContent = 'Error';
                }
            }
        }, this.config.HEALTH_CHECK_INTERVAL);
    }
    
    renderCleaningSummary(summary) {
        return `
            <div class="cleaning-summary">
                <h5>Cleaning Summary</h5>
                <pre>${JSON.stringify(summary, null, 2)}</pre>
            </div>
        `;
    }
    
    handleDataUpdate(data) {
        console.log('Data update received:', data);
        this.addActivityLog('Data updated', 'info');
    }
    
    updateSystemStatus(data) {
        console.log('System status update:', data);
    }
    
    makeBatchPrediction() {
        this.showAlert('Batch prediction feature coming soon!', 'info');
    }
    
    downloadModel() {
        this.showAlert('Model download feature coming soon!', 'info');
    }
}

const app = new MLPipelineApp();