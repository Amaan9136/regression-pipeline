class MLPipelineApp {
    constructor() {
        // Core application state
        this.socket = io();
        this.sessionId = null;
        this.currentDataset = null;
        this.trainingInProgress = false;
        this.currentStep = 'upload';
        
        // Data and results storage
        this.datasetInfo = null;
        this.cleaningOperations = {};
        this.modelResults = {};
        this.visualizations = {};
        this.predictionModels = [];
        
        // UI state
        this.activeTab = 'upload';
        this.healthCheckInterval = null;
        this.connectionEstablished = false;
        
        // Configuration
        this.config = {
            ALERT_AUTO_REMOVE_DELAY: 5000,
            MAX_ACTIVITY_LOGS: 50,
            HEALTH_CHECK_INTERVAL: 30000,
            PROGRESS_UPDATE_INTERVAL: 100
        };
        
        // Initialize application
        this.initialize();
    }
    
    // =================
    // INITIALIZATION
    // =================
    
    initialize() {
        this.setupEventListeners();
        this.setupWebSocketHandlers();
        this.initializeUI();
        this.startHealthCheck();
        console.log('🚀 ML Pipeline Application Initialized');
    }
    
    setupEventListeners() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.bindEventListeners());
        } else {
            this.bindEventListeners();
        }
    }
    
    bindEventListeners() {
        // Navigation tabs
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const section = e.target.dataset.section;
                if (section) this.switchSection(section);
            });
        });
        
        // File upload functionality
        this.setupFileUpload();
        
        // Data preprocessing handlers
        this.setupDataPreprocessing();
        
        // Training handlers
        this.setupTrainingHandlers();
        
        // Results and analytics
        this.setupResultsHandlers();
        
        // Prediction handlers
        this.setupPredictionHandlers();
        
        // Side panel handlers
        this.setupSidePanelHandlers();
        
        // Modal handlers
        this.setupModalHandlers();
        
        console.log('✅ Event listeners bound successfully');
    }
    
    // =================
    // WEBSOCKET SETUP
    // =================
    
    setupWebSocketHandlers() {
        // Connection events
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
        
        // Session management
        this.socket.on('session_created', (data) => {
            this.sessionId = data.session_id;
            this.updateSessionInfo();
            this.addActivityLog('Session created successfully', 'success');
            console.log(`📱 Session created: ${this.sessionId}`);
        });
        
        // Training progress
        this.socket.on('training_progress', (data) => {
            this.updateTrainingProgress(data);
            this.addActivityLog(data.message, 'info', data.progress);
            
            // Handle completion
            if (data.progress === 100 && data.data) {
                this.handleTrainingCompletion(data.data);
            }
        });
        
        // Training errors
        this.socket.on('training_error', (data) => {
            this.handleTrainingError(data.error);
        });
        
        // Real-time updates
        this.socket.on('data_update', (data) => {
            this.handleDataUpdate(data);
        });
        
        // System status updates
        this.socket.on('system_status', (data) => {
            this.updateSystemStatus(data);
        });
    }
    
    // =================
    // FILE UPLOAD SETUP
    // =================
    
    setupFileUpload() {
        const fileInput = document.getElementById('file-input');
        const uploadZone = document.getElementById('upload-zone');
        const datasetSelect = document.getElementById('dataset-select');
        
        if (uploadZone) {
            // Drag and drop functionality
            uploadZone.addEventListener('dragover', this.handleDragOver.bind(this));
            uploadZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
            uploadZone.addEventListener('drop', this.handleFileDrop.bind(this));
            uploadZone.addEventListener('click', () => fileInput?.click());
        }
        
        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                this.processFiles(Array.from(e.target.files));
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
        e.currentTarget.classList.add('dragover');
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
        const files = Array.from(e.dataTransfer.files);
        this.processFiles(files);
    }
    
    async processFiles(files) {
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
        try {
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
            } else {
                throw new Error(result.error || 'Upload failed');
            }
        } catch (error) {
            this.showAlert(`Upload failed: ${error.message}`, 'error');
            console.error('Upload error:', error);
        } finally {
            this.hideLoading();
        }
    }
    
    async selectDataset(filename) {
        if (!filename) return;
        
        try {
            this.showLoading('Loading dataset information...');
            
            // Get dataset info
            const response = await fetch(`/api/dataset_info/${filename}`);
            const result = await response.json();
            
            if (result.success) {
                this.currentDataset = filename;
                this.datasetInfo = result.info;
                this.displayDatasetPreview(result.info);
                this.updateDatasetStats();
                this.addActivityLog(`Dataset selected: ${filename}`, 'info');
            } else {
                throw new Error(result.error || 'Failed to load dataset info');
            }
        } catch (error) {
            this.showAlert(`Failed to load dataset: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    // =================
    // DATA PREPROCESSING
    // =================
    
    setupDataPreprocessing() {
        // Clean data button
        const cleanDataBtn = document.getElementById('clean-data-btn');
        if (cleanDataBtn) {
            cleanDataBtn.addEventListener('click', () => this.showCleaningInterface());
        }
        
        // Apply cleaning button
        const applyCleaningBtn = document.getElementById('apply-cleaning-btn');
        if (applyCleaningBtn) {
            applyCleaningBtn.addEventListener('click', () => this.applyDataCleaning());
        }
        
        // Detect outliers button
        const detectOutliersBtn = document.getElementById('detect-outliers-btn');
        if (detectOutliersBtn) {
            detectOutliersBtn.addEventListener('click', () => this.detectOutliers());
        }
        
        // Tab switching for cleaning
        document.querySelectorAll('.tab-btn').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabName = e.target.dataset.tab;
                if (tabName) this.switchCleaningTab(tabName);
            });
        });
    }
    
    async showCleaningInterface() {
        if (!this.currentDataset) {
            this.showAlert('Please select a dataset first', 'warning');
            return;
        }
        
        this.switchSection('preprocessing');
        await this.loadDatasetAnalysis();
    }
    
    async loadDatasetAnalysis() {
        try {
            this.showLoading('Analyzing dataset...');
            
            const response = await fetch(`/api/analyze_dataset/${this.currentDataset}`);
            const result = await response.json();
            
            if (result.success) {
                this.displayDataAnalysis(result.analysis);
                this.setupCleaningControls(result.analysis);
            } else {
                throw new Error(result.error || 'Analysis failed');
            }
        } catch (error) {
            this.showAlert(`Analysis failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    displayDataAnalysis(analysis) {
        // Display overview
        this.displayDataOverview(analysis);
        
        // Display column information
        this.displayColumnAnalysis(analysis.columns);
        
        // Display missing values
        this.displayMissingValues(analysis.missing_values);
        
        // Display statistical summary
        this.displayStatisticalSummary(analysis.statistics);
    }
    
    displayDataOverview(analysis) {
        const overviewDiv = document.getElementById('data-overview');
        if (!overviewDiv) return;
        
        overviewDiv.innerHTML = `
            <div class="overview-section">
                <h4>📊 Dataset Summary</h4>
                <div class="overview-grid">
                    <div class="overview-item">
                        <div class="overview-item-label">Total Rows</div>
                        <div class="overview-item-value">${analysis.shape[0].toLocaleString()}</div>
                    </div>
                    <div class="overview-item">
                        <div class="overview-item-label">Total Columns</div>
                        <div class="overview-item-value">${analysis.shape[1]}</div>
                    </div>
                    <div class="overview-item">
                        <div class="overview-item-label">Memory Usage</div>
                        <div class="overview-item-value">${this.formatBytes(analysis.memory_usage || 0)}</div>
                    </div>
                    <div class="overview-item">
                        <div class="overview-item-label">Missing Values</div>
                        <div class="overview-item-value">${analysis.total_missing || 0}</div>
                    </div>
                </div>
            </div>
            
            <div class="overview-section">
                <h4>🏷️ Data Types</h4>
                <div class="overview-grid">
                    <div class="overview-item">
                        <div class="overview-item-label">Numeric Columns</div>
                        <div class="overview-item-value">${analysis.numeric_columns?.length || 0}</div>
                    </div>
                    <div class="overview-item">
                        <div class="overview-item-label">Categorical Columns</div>
                        <div class="overview-item-value">${analysis.categorical_columns?.length || 0}</div>
                    </div>
                    <div class="overview-item">
                        <div class="overview-item-label">Boolean Columns</div>
                        <div class="overview-item-value">${analysis.boolean_columns?.length || 0}</div>
                    </div>
                    <div class="overview-item">
                        <div class="overview-item-label">DateTime Columns</div>
                        <div class="overview-item-value">${analysis.datetime_columns?.length || 0}</div>
                    </div>
                </div>
            </div>
        `;
    }
    
    async detectOutliers() {
        if (!this.currentDataset) {
            this.showAlert('Please select a dataset first', 'warning');
            return;
        }
        
        try {
            this.showLoading('Detecting outliers...');
            
            const method = document.getElementById('outlier-method')?.value || 'iqr';
            const threshold = document.getElementById('outlier-threshold')?.value || 1.5;
            
            const response = await fetch(`/api/detect_outliers/${this.currentDataset}?method=${method}&threshold=${threshold}`);
            const result = await response.json();
            
            if (result.success) {
                this.displayOutliers(result.outliers);
                this.addActivityLog('Outliers detected successfully', 'info');
            } else {
                throw new Error(result.error || 'Outlier detection failed');
            }
        } catch (error) {
            this.showAlert(`Outlier detection failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    displayOutliers(outliers) {
        const outliersDiv = document.getElementById('outliers-detector');
        if (!outliersDiv) return;
        
        let html = '<div class="outliers-summary"><h4>🎯 Outliers Detection Results</h4>';
        
        Object.entries(outliers).forEach(([column, data]) => {
            if (data.outliers && data.outliers.length > 0) {
                html += `
                    <div class="outlier-column">
                        <h5>${column}</h5>
                        <p>Found ${data.outliers.length} outliers</p>
                        <div class="outlier-actions">
                            <button class="btn btn-sm btn-secondary" onclick="mlApp.viewColumnOutliers('${column}')">
                                View Details
                            </button>
                            <button class="btn btn-sm btn-danger" onclick="mlApp.removeColumnOutliers('${column}')">
                                Remove Outliers
                            </button>
                        </div>
                    </div>
                `;
            }
        });
        
        html += '</div>';
        outliersDiv.innerHTML = html;
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
                this.showAlert('Data cleaning applied successfully!', 'success');
                this.displayCleaningResults(result);
                this.addActivityLog('Data cleaning completed', 'success');
                
                // Update current dataset to cleaned version
                if (result.cleaned_filename) {
                    this.currentDataset = result.cleaned_filename;
                }
            } else {
                throw new Error(result.error || 'Cleaning failed');
            }
        } catch (error) {
            this.showAlert(`Data cleaning failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    displayCleaningResults(results) {
        const resultsDiv = document.getElementById('cleaning-results');
        if (!resultsDiv) return;
        
        resultsDiv.innerHTML = `
            <div class="cleaning-summary">
                <h4>🧹 Cleaning Results</h4>
                <div class="results-grid">
                    <div class="result-item">
                        <span class="result-label">Original Shape:</span>
                        <span class="result-value">${results.original_shape[0]} × ${results.original_shape[1]}</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Final Shape:</span>
                        <span class="result-value">${results.final_shape[0]} × ${results.final_shape[1]}</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Cleaned File:</span>
                        <span class="result-value">${results.cleaned_filename}</span>
                    </div>
                </div>
                ${results.cleaning_summary ? this.renderCleaningSummary(results.cleaning_summary) : ''}
            </div>
        `;
    }
    
    // =================
    // TRAINING SETUP
    // =================
    
    setupTrainingHandlers() {
        // Start training button
        const startTrainingBtn = document.getElementById('start-training');
        if (startTrainingBtn) {
            startTrainingBtn.addEventListener('click', () => this.startTraining());
        }
        
        // Stop training button
        const stopTrainingBtn = document.getElementById('stop-training');
        if (stopTrainingBtn) {
            stopTrainingBtn.addEventListener('click', () => this.stopTraining());
        }
        
        // Configuration updates
        this.setupTrainingConfiguration();
    }
    
    setupTrainingConfiguration() {
        // Training configuration form
        const configForm = document.getElementById('training-config-form');
        if (configForm) {
            configForm.addEventListener('change', () => this.updateTrainingConfig());
        }
        
        // Advanced configuration toggles
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
            this.showAlert('Training is already in progress', 'info');
            return;
        }
        
        try {
            this.trainingInProgress = true;
            this.updateTrainingUI(true);
            this.updateTrainingConfig();
            
            this.socket.emit('start_training', {
                session_id: this.sessionId,
                dataset: this.currentDataset,
                config: this.trainingConfig || {}
            });
            
            this.addActivityLog('Training started...', 'info');
            this.switchSection('training');
            
        } catch (error) {
            this.showAlert(`Error starting training: ${error.message}`, 'error');
            this.trainingInProgress = false;
            this.updateTrainingUI(false);
        }
    }
    
    stopTraining() {
        if (!this.trainingInProgress) {
            this.showAlert('No training in progress', 'info');
            return;
        }
        
        // Emit stop training signal
        this.socket.emit('stop_training', {
            session_id: this.sessionId
        });
        
        this.trainingInProgress = false;
        this.updateTrainingUI(false);
        this.addActivityLog('Training stopped by user', 'warning');
        this.showAlert('Training stopped', 'info');
    }
    
    updateTrainingProgress(data) {
        // Update progress bar
        this.updateProgressBar(data.progress, data.message);
        
        // Update training status
        const statusDiv = document.getElementById('training-status');
        if (statusDiv) {
            statusDiv.innerHTML = `
                <div class="status-item">
                    <span class="status-label">Progress:</span>
                    <span class="status-value">${data.progress}%</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Current Step:</span>
                    <span class="status-value">${data.message}</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Time:</span>
                    <span class="status-value">${new Date().toLocaleTimeString()}</span>
                </div>
            `;
        }
        
        // Update live metrics if available
        if (data.data && data.data.live_metrics) {
            this.updateLiveMetrics(data.data.live_metrics);
        }
    }
    
    handleTrainingCompletion(data) {
        this.trainingInProgress = false;
        this.updateTrainingUI(false);
        
        // Store results
        this.modelResults = data.model_summary || {};
        this.visualizations = data.plots || {};
        
        // Switch to results section
        this.switchSection('results');
        
        // Display results
        this.displayTrainingResults(data);
        
        // Update prediction models list
        this.updatePredictionModels();
        
        this.showAlert('Training completed successfully!', 'success');
        this.addActivityLog('Training completed successfully', 'success');
    }
    
    handleTrainingError(error) {
        this.trainingInProgress = false;
        this.updateTrainingUI(false);
        this.showAlert(`Training failed: ${error}`, 'error');
        this.addActivityLog(`Training failed: ${error}`, 'error');
    }
    
    // =================
    // RESULTS DISPLAY
    // =================
    
    setupResultsHandlers() {
        // Export results button
        const exportBtn = document.getElementById('export-results-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportResults());
        }
        
        // Model comparison
        const compareBtn = document.getElementById('compare-models-btn');
        if (compareBtn) {
            compareBtn.addEventListener('click', () => this.compareModels());
        }
        
        // Download model
        const downloadBtn = document.getElementById('download-model-btn');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => this.downloadModel());
        }
    }
    
    displayTrainingResults(data) {
        // Display model summary table
        this.displayModelSummary(data.model_summary);
        
        // Display visualizations
        this.displayVisualizations(data.plots);
        
        // Display feature importance
        if (data.feature_importance) {
            this.displayFeatureImportance(data.feature_importance);
        }
        
        // Display best model details
        this.displayBestModelDetails(data.model_summary);
        
        // Update statistics
        this.updateResultsStatistics(data);
    }
    
    displayModelSummary(modelSummary) {
        const tableBody = document.getElementById('models-tbody');
        if (!tableBody) return;
        
        const sortedModels = Object.entries(modelSummary)
            .sort(([,a], [,b]) => (b.r2 || 0) - (a.r2 || 0));
        
        tableBody.innerHTML = sortedModels.map(([model, metrics], index) => `
            <tr class="${index === 0 ? 'best-model' : ''}">
                <td>
                    <div class="model-name">
                        ${index === 0 ? '🏆 ' : ''}${model}
                        ${index === 0 ? '<span class="best-badge">Best</span>' : ''}
                    </div>
                </td>
                <td><span class="metric-value">${this.formatMetric(metrics.r2)}</span></td>
                <td><span class="metric-value">${this.formatMetric(metrics.mse)}</span></td>
                <td><span class="metric-value">${this.formatMetric(metrics.mae)}</span></td>
                <td><span class="metric-value">${this.formatMetric(metrics.rmse)}</span></td>
                <td><span class="metric-value">${this.formatMetric(metrics.cv_score)}</span></td>
                <td>
                    <div class="model-actions">
                        <button class="btn btn-sm btn-primary" onclick="mlApp.showModelDetails('${model}')">
                            📊 Details
                        </button>
                        <button class="btn btn-sm btn-secondary" onclick="mlApp.useModelForPrediction('${model}')">
                            🎯 Use for Prediction
                        </button>
                    </div>
                </td>
            </tr>
        `).join('');
    }
    
    displayVisualizations(plots) {
        const plotsContainer = document.getElementById('visualizations-container');
        if (!plotsContainer) return;
        
        let html = '';
        
        Object.entries(plots).forEach(([plotType, plotData]) => {
            html += `
                <div class="visualization-card">
                    <div class="viz-header">
                        <h4>${this.formatPlotTitle(plotType)}</h4>
                        <div class="viz-actions">
                            <button class="btn btn-sm btn-secondary" onclick="mlApp.downloadPlot('${plotType}')">
                                💾 Download
                            </button>
                            <button class="btn btn-sm btn-primary" onclick="mlApp.expandPlot('${plotType}')">
                                🔍 Expand
                            </button>
                        </div>
                    </div>
                    <div class="viz-content" id="plot-${plotType}">
                        ${this.renderPlot(plotType, plotData)}
                    </div>
                </div>
            `;
        });
        
        plotsContainer.innerHTML = html;
    }
    
    renderPlot(plotType, plotData) {
        // Handle different plot types
        if (plotData.type === 'plotly') {
            // Render Plotly plots
            setTimeout(() => {
                Plotly.newPlot(`plot-${plotType}`, plotData.data, plotData.layout, {responsive: true});
            }, 100);
            return '<div class="plotly-container"></div>';
        } else if (plotData.type === 'image') {
            // Render image plots
            return `<img src="data:image/png;base64,${plotData.data}" alt="${plotType}" class="plot-image">`;
        } else {
            // Handle other plot types
            return '<div class="plot-placeholder">Plot will be rendered here</div>';
        }
    }
    
    async exportResults() {
        if (!this.currentDataset) {
            this.showAlert('No results to export', 'warning');
            return;
        }
        
        try {
            this.showLoading('Exporting results...');
            
            const response = await fetch(`/api/export_results/${this.currentDataset.replace('.csv', '')}`);
            const result = await response.json();
            
            if (result.success) {
                this.showAlert('Results exported successfully!', 'success');
                this.addActivityLog('Results exported', 'success');
                
                // Trigger download if available
                if (result.download_url) {
                    window.open(result.download_url, '_blank');
                }
            } else {
                throw new Error(result.error || 'Export failed');
            }
        } catch (error) {
            this.showAlert(`Export failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    // =================
    // PREDICTION SETUP
    // =================
    
    setupPredictionHandlers() {
        // Model selection for prediction
        const modelSelect = document.getElementById('prediction-model-select');
        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => {
                this.selectPredictionModel(e.target.value);
            });
        }
        
        // Single prediction
        const predictBtn = document.getElementById('make-prediction-btn');
        if (predictBtn) {
            predictBtn.addEventListener('click', () => this.makeSinglePrediction());
        }
        
        // Batch prediction
        const batchPredictBtn = document.getElementById('batch-prediction-btn');
        if (batchPredictBtn) {
            batchPredictBtn.addEventListener('click', () => this.makeBatchPrediction());
        }
        
        // Clear predictions
        const clearPredictionsBtn = document.getElementById('clear-predictions-btn');
        if (clearPredictionsBtn) {
            clearPredictionsBtn.addEventListener('click', () => this.clearPredictions());
        }
    }
    
    updatePredictionModels() {
        const modelSelect = document.getElementById('prediction-model-select');
        if (!modelSelect) return;
        
        // Clear existing options
        modelSelect.innerHTML = '<option value="">Select a model...</option>';
        
        // Add trained models
        Object.keys(this.modelResults).forEach(modelName => {
            const option = document.createElement('option');
            option.value = modelName;
            option.textContent = modelName;
            modelSelect.appendChild(option);
        });
        
        // Update prediction interface visibility
        const predictionInterface = document.getElementById('prediction-interface');
        if (predictionInterface) {
            predictionInterface.style.display = Object.keys(this.modelResults).length > 0 ? 'block' : 'none';
        }
    }
    
    selectPredictionModel(modelName) {
        if (!modelName) return;
        
        this.selectedPredictionModel = modelName;
        this.setupPredictionForm(modelName);
        this.addActivityLog(`Selected model for prediction: ${modelName}`, 'info');
    }
    
    setupPredictionForm(modelName) {
        const formContainer = document.getElementById('prediction-form-container');
        if (!formContainer || !this.datasetInfo) return;
        
        // Get feature columns (exclude target column if known)
        const features = this.datasetInfo.columns.filter(col => 
            col !== 'target' && col !== 'Target' && !col.toLowerCase().includes('target')
        );
        
        let formHTML = '<div class="prediction-form"><h4>Enter Feature Values</h4>';
        
        features.forEach(feature => {
            formHTML += `
                <div class="form-group">
                    <label for="feature-${feature}">${feature}:</label>
                    <input type="number" id="feature-${feature}" name="${feature}" 
                           class="form-control" step="any" required>
                </div>
            `;
        });
        
        formHTML += '</div>';
        formContainer.innerHTML = formHTML;
    }
    
    async makeSinglePrediction() {
        if (!this.selectedPredictionModel) {
            this.showAlert('Please select a model first', 'warning');
            return;
        }
        
        try {
            this.showLoading('Making prediction...');
            
            // Collect feature values
            const features = {};
            const formInputs = document.querySelectorAll('#prediction-form-container input');
            
            for (const input of formInputs) {
                if (!input.value) {
                    throw new Error(`Please enter a value for ${input.name}`);
                }
                features[input.name] = parseFloat(input.value);
            }
            
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_name: this.selectedPredictionModel,
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
    
    displayPredictionResult(result) {
        const resultsDiv = document.getElementById('prediction-results');
        if (!resultsDiv) return;
        
        const timestamp = new Date().toLocaleString();
        
        const resultHTML = `
            <div class="prediction-result">
                <div class="result-header">
                    <h4>🎯 Prediction Result</h4>
                    <span class="result-timestamp">${timestamp}</span>
                </div>
                <div class="result-content">
                    <div class="prediction-value">
                        <span class="value-label">Predicted Value:</span>
                        <span class="value-number">${result.prediction.toFixed(4)}</span>
                    </div>
                    <div class="prediction-details">
                        <div class="detail-item">
                            <span class="detail-label">Model Used:</span>
                            <span class="detail-value">${result.model_used}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Features Used:</span>
                            <span class="detail-value">${Object.keys(result.features_used).length}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        resultsDiv.insertAdjacentHTML('afterbegin', resultHTML);
    }
    
    async makeBatchPrediction() {
        const fileInput = document.getElementById('batch-prediction-file');
        if (!fileInput?.files[0]) {
            this.showAlert('Please select a file for batch prediction', 'warning');
            return;
        }
        
        if (!this.selectedPredictionModel) {
            this.showAlert('Please select a model first', 'warning');
            return;
        }
        
        try {
            this.showLoading('Processing batch predictions...');
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('model_name', this.selectedPredictionModel);
            
            const response = await fetch('/api/batch_predict', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.displayBatchPredictionResults(result);
                this.addActivityLog(`Batch prediction completed: ${result.predictions.length} predictions`, 'success');
            } else {
                throw new Error(result.error || 'Batch prediction failed');
            }
        } catch (error) {
            this.showAlert(`Batch prediction failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    displayBatchPredictionResults(result) {
        const resultsDiv = document.getElementById('batch-results');
        if (!resultsDiv) return;
        
        let html = `
            <div class="batch-prediction-results">
                <h4>📊 Batch Prediction Results</h4>
                <div class="batch-summary">
                    <span>Total Predictions: ${result.predictions.length}</span>
                    <span>Model Used: ${result.model_used}</span>
                </div>
        `;
        
        if (result.predictions.length <= 100) {
            // Show detailed results for smaller batches
            html += '<div class="predictions-table-container">';
            html += '<table class="predictions-table">';
            html += '<thead><tr><th>Row</th><th>Prediction</th></tr></thead><tbody>';
            
            result.predictions.forEach((pred, index) => {
                html += `<tr><td>${index + 1}</td><td>${pred.toFixed(4)}</td></tr>`;
            });
            
            html += '</tbody></table></div>';
        } else {
            // Show summary for larger batches
            const stats = this.calculatePredictionStats(result.predictions);
            html += `
                <div class="predictions-summary">
                    <div class="stat-item">
                        <span class="stat-label">Mean:</span>
                        <span class="stat-value">${stats.mean.toFixed(4)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Std Dev:</span>
                        <span class="stat-value">${stats.std.toFixed(4)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Min:</span>
                        <span class="stat-value">${stats.min.toFixed(4)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Max:</span>
                        <span class="stat-value">${stats.max.toFixed(4)}</span>
                    </div>
                </div>
            `;
        }
        
        html += `
                <div class="batch-actions">
                    <button class="btn btn-primary" onclick="mlApp.downloadBatchResults()">
                        💾 Download Results
                    </button>
                </div>
            </div>
        `;
        
        resultsDiv.innerHTML = html;
    }
    
    // =================
    // SIDE PANEL SETUP
    // =================
    
    setupSidePanelHandlers() {
        const panelToggle = document.getElementById('panel-toggle');
        if (panelToggle) {
            panelToggle.addEventListener('click', () => this.toggleSidePanel());
        }
    }
    
    toggleSidePanel() {
        const sidePanel = document.getElementById('side-panel');
        if (!sidePanel) return;
        
        sidePanel.classList.toggle('collapsed');
        const toggle = document.getElementById('panel-toggle');
        if (toggle) {
            toggle.textContent = sidePanel.classList.contains('collapsed') ? '▶' : '◀';
        }
    }
    
    // =================
    // MODAL SETUP
    // =================
    
    setupModalHandlers() {
        // Close modal handlers
        document.querySelectorAll('.modal-close').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const modal = e.target.closest('.modal');
                if (modal) this.closeModal(modal.id);
            });
        });
        
        // Click outside to close
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
    
    showModelDetails(modelName) {
        const modelData = this.modelResults[modelName];
        if (!modelData) return;
        
        const modalBody = document.getElementById('model-details-body');
        if (!modalBody) return;
        
        modalBody.innerHTML = `
            <div class="model-details">
                <h4>${modelName}</h4>
                <div class="metrics-grid">
                    ${Object.entries(modelData).map(([metric, value]) => `
                        <div class="metric-item">
                            <span class="metric-label">${this.formatMetricName(metric)}:</span>
                            <span class="metric-value">${this.formatMetric(value)}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        
        this.showModal('model-details-modal');
    }
    
    // =================
    // UTILITY FUNCTIONS
    // =================
    
    switchSection(sectionName) {
        // Update navigation
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-section="${sectionName}"]`)?.classList.add('active');
        
        // Update sections
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });
        document.getElementById(`${sectionName}-section`)?.classList.add('active');
        
        this.activeTab = sectionName;
        this.addActivityLog(`Switched to ${sectionName} section`, 'info');
    }
    
    switchCleaningTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`)?.classList.add('active');
        
        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`)?.classList.add('active');
    }
    
    updateProgressBar(progress, message) {
        const progressBar = document.querySelector('.progress-bar');
        const progressText = document.querySelector('.progress-text');
        const progressPercentage = document.querySelector('.progress-percentage');
        
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);
        }
        
        if (progressText) {
            progressText.textContent = message;
        }
        
        if (progressPercentage) {
            progressPercentage.textContent = `${progress}%`;
        }
    }
    
    updateTrainingUI(isTraining) {
        const startBtn = document.getElementById('start-training');
        const stopBtn = document.getElementById('stop-training');
        
        if (startBtn) {
            startBtn.style.display = isTraining ? 'none' : 'inline-flex';
            startBtn.disabled = isTraining;
        }
        
        if (stopBtn) {
            stopBtn.style.display = isTraining ? 'inline-flex' : 'none';
        }
        
        // Update progress section visibility
        const progressSection = document.getElementById('training-progress-section');
        if (progressSection) {
            progressSection.style.display = isTraining ? 'block' : 'none';
        }
    }
    
    showAlert(message, type = 'info') {
        const alertsContainer = document.getElementById('notifications') || document.body;
        
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type}`;
        alertDiv.innerHTML = `
            <div class="alert-content">
                <span class="alert-icon">${this.getAlertIcon(type)}</span>
                <span class="alert-message">${message}</span>
                <button class="alert-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        alertsContainer.appendChild(alertDiv);
        
        // Auto-remove after delay
        setTimeout(() => {
            if (alertDiv.parentElement) {
                alertDiv.remove();
            }
        }, this.config.ALERT_AUTO_REMOVE_DELAY);
        
        // Animate in
        setTimeout(() => alertDiv.classList.add('show'), 10);
    }
    
    getAlertIcon(type) {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        return icons[type] || icons.info;
    }
    
    showLoading(message = 'Processing...') {
        const loadingOverlay = document.getElementById('loading-overlay');
        const loadingMessage = document.getElementById('loading-message');
        const loadingTitle = document.getElementById('loading-title');
        
        if (loadingMessage) loadingMessage.textContent = message;
        if (loadingTitle) loadingTitle.textContent = 'Processing...';
        if (loadingOverlay) loadingOverlay.style.display = 'flex';
    }
    
    hideLoading() {
        const loadingOverlay = document.getElementById('loading-overlay');
        if (loadingOverlay) loadingOverlay.style.display = 'none';
    }
    
    addActivityLog(message, type = 'info', progress = null) {
        const activityList = document.getElementById('activity-list');
        if (!activityList) return;
        
        const timestamp = new Date().toLocaleTimeString();
        const logItem = document.createElement('div');
        logItem.className = `activity-item activity-${type}`;
        
        logItem.innerHTML = `
            <div class="activity-header">
                <span class="activity-icon">${this.getAlertIcon(type)}</span>
                <span class="activity-time">${timestamp}</span>
            </div>
            <div class="activity-message">${message}</div>
            ${progress !== null ? `<div class="activity-progress">${progress}%</div>` : ''}
        `;
        
        activityList.insertBefore(logItem, activityList.firstChild);
        
        // Limit activity logs
        while (activityList.children.length > this.config.MAX_ACTIVITY_LOGS) {
            activityList.removeChild(activityList.lastChild);
        }
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
            datasetsProcessed.textContent = '1'; // Current dataset
        }
    }
    
    displayDatasetPreview(preview) {
        const previewContainer = document.getElementById('dataset-preview');
        if (!previewContainer || !preview) return;
        
        let html = `
            <div class="dataset-info">
                <h4>📊 Dataset Preview</h4>
                <div class="dataset-stats">
                    <span class="stat-item">Rows: ${preview.shape[0]}</span>
                    <span class="stat-item">Columns: ${preview.shape[1]}</span>
                </div>
            </div>
        `;
        
        if (preview.sample_data && preview.sample_data.length > 0) {
            html += '<div class="preview-table-container">';
            html += '<table class="preview-table">';
            html += '<thead><tr>';
            
            preview.columns.forEach(col => {
                html += `<th>${col}</th>`;
            });
            
            html += '</tr></thead><tbody>';
            
            preview.sample_data.slice(0, 5).forEach(row => {
                html += '<tr>';
                preview.columns.forEach(col => {
                    html += `<td>${row[col] || '-'}</td>`;
                });
                html += '</tr>';
            });
            
            html += '</tbody></table></div>';
        }
        
        previewContainer.innerHTML = html;
    }
    
    // Health check system
    async startHealthCheck() {
        this.healthCheckInterval = setInterval(async () => {
            try {
                const response = await fetch('/health');
                const result = await response.json();
                
                if (result.status === 'healthy') {
                    // Update health indicator
                    const healthIndicator = document.getElementById('health-indicator');
                    if (healthIndicator) {
                        const statusDot = healthIndicator.querySelector('.status-dot');
                        const statusText = healthIndicator.querySelector('.status-text');
                        
                        if (statusDot) statusDot.className = 'status-dot status-healthy';
                        if (statusText) statusText.textContent = 'Healthy';
                    }
                    
                    // Update active sessions
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
    
    // Formatting utilities
    formatMetric(value) {
        if (value === null || value === undefined || isNaN(value)) {
            return 'N/A';
        }
        return typeof value === 'number' ? value.toFixed(4) : value;
    }
    
    formatMetricName(name) {
        const nameMap = {
            'r2': 'R² Score',
            'mse': 'MSE',
            'mae': 'MAE',
            'rmse': 'RMSE',
            'cv_score': 'CV Score',
            'cv_std': 'CV Std Dev'
        };
        return nameMap[name] || name.replace('_', ' ').toUpperCase();
    }
    
    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    formatPlotTitle(plotType) {
        const titleMap = {
            'correlation_heatmap': '🔥 Correlation Heatmap',
            'feature_importance': '⭐ Feature Importance',
            'residuals_plot': '📈 Residuals Analysis',
            'prediction_vs_actual': '🎯 Predictions vs Actual',
            'model_comparison': '📊 Model Comparison',
            'learning_curves': '📈 Learning Curves'
        };
        return titleMap[plotType] || plotType.replace('_', ' ').toUpperCase();
    }
    
    calculatePredictionStats(predictions) {
        const mean = predictions.reduce((a, b) => a + b, 0) / predictions.length;
        const variance = predictions.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / predictions.length;
        const std = Math.sqrt(variance);
        const min = Math.min(...predictions);
        const max = Math.max(...predictions);
        
        return { mean, std, min, max };
    }
    
    // Initialize UI components
    initializeUI() {
        // Set up uptime counter
        this.startTime = Date.now();
        setInterval(() => {
            const uptimeSpan = document.getElementById('uptime');
            if (uptimeSpan) {
                const uptime = Math.floor((Date.now() - this.startTime) / 1000);
                const minutes = Math.floor(uptime / 60);
                const seconds = uptime % 60;
                uptimeSpan.textContent = `${minutes}m ${seconds}s`;
            }
        }, 1000);
        
        // Initialize tooltips and help text
        this.initializeTooltips();
        
        // Set initial UI state
        this.switchSection('upload');
    }
    
    initializeTooltips() {
        // Add tooltips to metric headers and important elements
        const tooltips = {
            'r2': 'R-squared: Coefficient of determination (0-1, higher is better)',
            'mse': 'Mean Squared Error: Average squared differences (lower is better)', 
            'mae': 'Mean Absolute Error: Average absolute differences (lower is better)',
            'rmse': 'Root Mean Squared Error: Square root of MSE (lower is better)',
            'cv_score': 'Cross-validation Score: Average performance across folds'
        };
        
        Object.entries(tooltips).forEach(([key, text]) => {
            const elements = document.querySelectorAll(`[data-metric="${key}"]`);
            elements.forEach(el => {
                el.title = text;
            });
        });
    }
    
    // Cleanup on page unload
    cleanup() {
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
        }
        
        if (this.socket) {
            this.socket.disconnect();
        }
    }
}

let mlApp;
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        mlApp = new MLPipelineApp();
    });
} else {
    mlApp = new MLPipelineApp();
}

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (mlApp) {
        mlApp.cleanup();
    }
});

// Export for global access
window.mlApp = mlApp;

// Additional utility functions for global access
window.downloadPlot = function(plotType) {
    if (mlApp) {
        mlApp.downloadPlot(plotType);
    }
};

window.expandPlot = function(plotType) {
    if (mlApp) {
        mlApp.expandPlot(plotType);
    }
};

window.useModelForPrediction = function(modelName) {
    if (mlApp) {
        mlApp.useModelForPrediction(modelName);
        mlApp.switchSection('prediction');
    }
};