// Initialize Socket.IO connection
const socket = io();
let sessionId = null;
let currentDataset = null;
let cleaningOperations = {};

// Constants
const ALERT_AUTO_REMOVE_DELAY = 5000;
const MAX_ACTIVITY_LOGS = 50;

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    setupWebSocket();
    initializeFileUpload();
});

function initializeEventListeners() {
    // File upload elements
    const fileInput = document.getElementById('file-input');
    const uploadZone = document.getElementById('upload-zone');
    const datasetSelect = document.getElementById('dataset-select');
    
    if (uploadZone) {
        // Drag and drop functionality
        uploadZone.addEventListener('dragover', handleDragOver);
        uploadZone.addEventListener('dragleave', handleDragLeave);
        uploadZone.addEventListener('drop', handleFileDrop);
        uploadZone.addEventListener('click', () => fileInput?.click());
    }
    
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    // Dataset selection
    if (datasetSelect) {
        datasetSelect.addEventListener('change', handleDatasetSelection);
    }
    
    // Action buttons
    const cleanDataBtn = document.getElementById('clean-data-btn');
    const trainModelsBtn = document.getElementById('train-models-btn');
    const applyCleaningBtn = document.getElementById('apply-cleaning-btn');
    const exportResultsBtn = document.getElementById('export-results-btn');
    
    if (cleanDataBtn) {
        cleanDataBtn.addEventListener('click', showCleaningInterface);
    }
    
    if (trainModelsBtn) {
        trainModelsBtn.addEventListener('click', startTraining);
    }
    
    if (applyCleaningBtn) {
        applyCleaningBtn.addEventListener('click', applyCleaningOperations);
    }
    
    if (exportResultsBtn) {
        exportResultsBtn.addEventListener('click', exportResults);
    }
    
    // Cleaning tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });
}

function setupWebSocket() {
    socket.on('session_created', (data) => {
        sessionId = data.session_id;
        addActivityLog('Session created successfully', 0);
        updateConnectionStatus(true);
    });

    socket.on('training_progress', (data) => {
        updateProgress(data.progress, data.message);
        addActivityLog(data.message, data.progress);
        
        if (data.data && data.progress === 100) {
            displayTrainingResults(data.data);
        }
    });

    socket.on('training_error', (data) => {
        showAlert('Training failed: ' + data.error, 'error');
        hideLoading();
        updateProgress(0, 'Training failed');
    });

    socket.on('connect', () => {
        console.log('Connected to server');
        updateConnectionStatus(true);
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        updateConnectionStatus(false);
        showAlert('Connection lost. Please refresh the page.', 'warning');
    });
}

function initializeFileUpload() {
    // Initialize drag and drop styling
    const uploadZone = document.getElementById('upload-zone');
    if (uploadZone) {
        uploadZone.classList.add('upload-ready');
    }
}

// File handling functions
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('dragover');
}

function handleFileDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files);
    processFiles(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    processFiles(files);
}

async function processFiles(files) {
    const csvFiles = files.filter(file => file.name.endsWith('.csv'));
    
    if (csvFiles.length === 0) {
        showAlert('Please select CSV files only.', 'warning');
        return;
    }

    for (const file of csvFiles) {
        await uploadFile(file);
    }
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        showLoading(`Uploading ${file.name}...`);
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        
        if (result.success) {
            showAlert(`File "${file.name}" uploaded successfully!`, 'success');
            addDatasetToSelect(result.filename);
            addActivityLog(`Uploaded: ${file.name}`, 100);
            
            // Auto-select the uploaded file
            const datasetSelect = document.getElementById('dataset-select');
            if (datasetSelect) {
                datasetSelect.value = result.filename;
                currentDataset = result.filename;
                await loadDatasetInfo(result.filename);
            }
        } else {
            showAlert(`Upload failed: ${result.error}`, 'error');
        }
    } catch (error) {
        showAlert(`Upload error: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

function addDatasetToSelect(filename) {
    const select = document.getElementById('dataset-select');
    if (!select) return;
    
    // Check if option already exists
    const existingOption = Array.from(select.options).find(option => option.value === filename);
    if (existingOption) return;
    
    const option = document.createElement('option');
    option.value = filename;
    option.textContent = filename;
    select.appendChild(option);
}

async function handleDatasetSelection(e) {
    const filename = e.target.value;
    if (!filename) return;
    
    currentDataset = filename;
    await loadDatasetInfo(filename);
    addActivityLog(`Selected dataset: ${filename}`, 0);
}

async function loadDatasetInfo(filename) {
    try {
        showLoading('Loading dataset information...');
        
        const response = await fetch(`/api/dataset_info/${filename}`);
        const result = await response.json();
        
        if (result.success) {
            displayDatasetInfo(result.dataset_info);
            enableActionButtons();
        } else {
            showAlert(`Failed to load dataset info: ${result.error}`, 'error');
        }
    } catch (error) {
        showAlert(`Error loading dataset: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

function displayDatasetInfo(datasetInfo) {
    // Update overview tab
    const overviewTab = document.getElementById('overview-tab');
    if (overviewTab) {
        overviewTab.innerHTML = `
            <div class="dataset-overview">
                <h3>Dataset Overview</h3>
                <div class="overview-grid">
                    <div class="overview-item">
                        <span class="label">Filename:</span>
                        <span class="value">${datasetInfo.filename}</span>
                    </div>
                    <div class="overview-item">
                        <span class="label">Shape:</span>
                        <span class="value">${datasetInfo.shape[0]} rows × ${datasetInfo.shape[1]} columns</span>
                    </div>
                    <div class="overview-item">
                        <span class="label">Memory Usage:</span>
                        <span class="value">${datasetInfo.memory_usage.toFixed(2)} MB</span>
                    </div>
                    <div class="overview-item">
                        <span class="label">Quality Score:</span>
                        <span class="value">${datasetInfo.quality_report.quality_score.toFixed(1)}%</span>
                    </div>
                </div>
                
                <h4>Missing Values</h4>
                <div class="missing-values-summary">
                    ${Object.keys(datasetInfo.missing_values).length > 0 ? 
                        Object.entries(datasetInfo.missing_values).map(([col, info]) => 
                            `<div class="missing-item">
                                <span class="column">${col}:</span>
                                <span class="count">${info.count} (${info.percentage.toFixed(1)}%)</span>
                            </div>`
                        ).join('') : 
                        '<p>No missing values found</p>'
                    }
                </div>
                
                <h4>Data Types</h4>
                <div class="dtypes-summary">
                    ${Object.entries(datasetInfo.dtypes).map(([col, dtype]) => 
                        `<div class="dtype-item">
                            <span class="column">${col}:</span>
                            <span class="dtype">${dtype}</span>
                        </div>`
                    ).join('')}
                </div>
            </div>
        `;
    }
    
    // Update columns tab
    displayColumnInfo(datasetInfo.columns);
    
    // Update missing values tab
    displayMissingValuesInfo(datasetInfo.missing_values);
}

function displayColumnInfo(columns) {
    const columnsList = document.getElementById('columns-list');
    if (!columnsList) return;
    
    columnsList.innerHTML = Object.entries(columns).map(([col, info]) => `
        <div class="column-item">
            <div class="column-header">
                <h4>${col}</h4>
                <span class="column-type">${info.dtype}</span>
            </div>
            <div class="column-details">
                <p><strong>Missing:</strong> ${info.missing_count} (${info.missing_percentage.toFixed(1)}%)</p>
                <p><strong>Unique values:</strong> ${info.unique_values}</p>
                ${info.min !== undefined ? `
                    <p><strong>Range:</strong> ${info.min.toFixed(2)} - ${info.max.toFixed(2)}</p>
                    <p><strong>Mean:</strong> ${info.mean.toFixed(2)} ± ${info.std.toFixed(2)}</p>
                ` : ''}
                <p><strong>Sample values:</strong> ${info.sample_values.join(', ')}</p>
            </div>
        </div>
    `).join('');
}

function displayMissingValuesInfo(missingValues) {
    const missingContent = document.getElementById('missing-values-content');
    if (!missingContent) return;
    
    if (Object.keys(missingValues).length === 0) {
        missingContent.innerHTML = '<p class="no-missing">No missing values found in the dataset.</p>';
        return;
    }
    
    missingContent.innerHTML = `
        <div class="missing-values-interface">
            <h3>Handle Missing Values</h3>
            ${Object.entries(missingValues).map(([col, info]) => `
                <div class="missing-column">
                    <h4>${col}</h4>
                    <p>Missing: ${info.count} values (${info.percentage.toFixed(1)}%)</p>
                    <select class="missing-strategy" data-column="${col}">
                        <option value="drop">Drop rows</option>
                        <option value="mean">Fill with mean</option>
                        <option value="median">Fill with median</option>
                        <option value="mode">Fill with mode</option>
                        <option value="forward_fill">Forward fill</option>
                        <option value="backward_fill">Backward fill</option>
                    </select>
                </div>
            `).join('')}
        </div>
    `;
}

function enableActionButtons() {
    const cleanDataBtn = document.getElementById('clean-data-btn');
    const trainModelsBtn = document.getElementById('train-models-btn');
    
    if (cleanDataBtn) {
        cleanDataBtn.disabled = false;
        cleanDataBtn.classList.remove('disabled');
    }
    
    if (trainModelsBtn) {
        trainModelsBtn.disabled = false;
        trainModelsBtn.classList.remove('disabled');
    }
}

function showCleaningInterface() {
    const cleaningInterface = document.getElementById('cleaning-interface');
    if (cleaningInterface) {
        cleaningInterface.style.display = 'block';
        cleaningInterface.scrollIntoView({ behavior: 'smooth' });
    }
    addActivityLog('Opened data cleaning interface', 0);
}

async function applyCleaningOperations() {
    if (!currentDataset) {
        showAlert('Please select a dataset first', 'warning');
        return;
    }
    
    try {
        showLoading('Applying cleaning operations...');
        
        // Collect cleaning operations from UI
        const operations = {};
        
        // Get missing value strategies
        const missingStrategies = {};
        document.querySelectorAll('.missing-strategy').forEach(select => {
            const column = select.dataset.column;
            const method = select.value;
            if (method !== 'none') {
                missingStrategies[column] = { method: method };
            }
        });
        
        if (Object.keys(missingStrategies).length > 0) {
            operations.missing_values = missingStrategies;
        }
        
        const response = await fetch('/api/apply_cleaning', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filename: currentDataset,
                operations: operations
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            showAlert('Data cleaning applied successfully!', 'success');
            addActivityLog(`Cleaned data saved as: ${result.cleaned_filename}`, 100);
            
            // Add cleaned dataset to select
            addDatasetToSelect(result.cleaned_filename);
            
            // Update UI with results
            displayCleaningResults(result);
        } else {
            showAlert(`Cleaning failed: ${result.error}`, 'error');
        }
    } catch (error) {
        showAlert(`Error applying cleaning: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

function displayCleaningResults(results) {
    const resultHtml = `
        <div class="cleaning-results">
            <h3>Cleaning Results</h3>
            <p><strong>Original shape:</strong> ${results.original_shape[0]} × ${results.original_shape[1]}</p>
            <p><strong>Final shape:</strong> ${results.final_shape[0]} × ${results.final_shape[1]}</p>
            <p><strong>Quality score:</strong> ${results.quality_report.quality_score.toFixed(1)}%</p>
            
            ${results.results.missing_values_handled ? `
                <h4>Missing Values Handled:</h4>
                <ul>
                    ${Object.entries(results.results.missing_values_handled).map(([col, info]) => 
                        `<li>${col}: ${info}</li>`
                    ).join('')}
                </ul>
            ` : ''}
        </div>
    `;
    
    // Display results in a modal or dedicated area
    showAlert('Cleaning completed successfully!', 'success');
}

async function startTraining() {
    if (!currentDataset) {
        showAlert('Please select a dataset first', 'warning');
        return;
    }
    
    if (!sessionId) {
        showAlert('Session not established. Please refresh the page.', 'error');
        return;
    }
    
    try {
        showLoading('Starting model training...');
        
        const config = {
            test_size: 0.2,
            random_state: 42,
            cross_validation_folds: 5,
            hyperparameter_tuning: true,
            feature_engineering: true
        };
        
        socket.emit('start_training', {
            session_id: sessionId,
            dataset: currentDataset,
            config: config
        });
        
        addActivityLog('Training started...', 0);
        
    } catch (error) {
        showAlert(`Error starting training: ${error.message}`, 'error');
        hideLoading();
    }
}

function displayTrainingResults(data) {
    hideLoading();
    
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) {
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Display model summary
    if (data.model_summary) {
        displayModelSummary(data.model_summary);
    }
    
    // Display visualizations
    if (data.plots) {
        displayPlots(data.plots);
    }
    
    addActivityLog('Training completed successfully!', 100);
    showAlert('Model training completed successfully!', 'success');
}

function displayModelSummary(modelSummary) {
    const modelsTable = document.getElementById('models-tbody');
    if (!modelsTable) return;
    
    modelsTable.innerHTML = Object.entries(modelSummary).map(([model, metrics]) => `
        <tr>
            <td><strong>${model}</strong></td>
            <td>${metrics.r2 ? metrics.r2.toFixed(4) : 'N/A'}</td>
            <td>${metrics.mse ? metrics.mse.toFixed(4) : 'N/A'}</td>
            <td>${metrics.mae ? metrics.mae.toFixed(4) : 'N/A'}</td>
            <td>${metrics.rmse ? metrics.rmse.toFixed(4) : 'N/A'}</td>
            <td>${metrics.cv_score ? `${metrics.cv_score.toFixed(4)} ± ${metrics.cv_std.toFixed(4)}` : 'N/A'}</td>
            <td>
                <button class="btn btn-sm btn-info" onclick="showModelDetails('${model}')">
                    Details
                </button>
            </td>
        </tr>
    `).join('');
}

function displayPlots(plots) {
    const visualizationGrid = document.getElementById('visualization-grid');
    if (!visualizationGrid) return;
    
    visualizationGrid.innerHTML = Object.entries(plots).map(([plotName, plotPath]) => `
        <div class="visualization-card">
            <h4>${plotName.replace('_', ' ').toUpperCase()}</h4>
            <img src="${plotPath}" alt="${plotName}" class="plot-image">
        </div>
    `).join('');
}

async function exportResults() {
    if (!currentDataset) {
        showAlert('No training results to export', 'warning');
        return;
    }
    
    try {
        showLoading('Exporting results...');
        
        const datasetName = currentDataset.replace('.csv', '');
        const response = await fetch(`/api/export_results/${datasetName}`);
        const result = await response.json();
        
        if (result.success) {
            showAlert('Results exported successfully!', 'success');
            addActivityLog('Results exported', 100);
        } else {
            showAlert(`Export failed: ${result.error}`, 'error');
        }
    } catch (error) {
        showAlert(`Export error: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

function switchTab(tabName) {
    // Remove active class from all tabs and contents
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    // Add active class to selected tab and content
    const selectedTab = document.querySelector(`[data-tab="${tabName}"]`);
    const selectedContent = document.getElementById(`${tabName}-tab`);
    
    if (selectedTab) selectedTab.classList.add('active');
    if (selectedContent) selectedContent.classList.add('active');
}

// Progress and UI update functions
function updateProgress(progress, message) {
    const progressBar = document.querySelector('.progress-bar');
    const progressText = document.querySelector('.progress-text');
    
    if (progressBar) {
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
    }
    
    if (progressText) {
        progressText.textContent = `${progress}% - ${message}`;
    }
}

function addActivityLog(message, progress) {
    const activityStream = document.getElementById('activity-stream');
    if (!activityStream) return;
    
    const timestamp = new Date().toLocaleTimeString();
    const logItem = document.createElement('div');
    logItem.className = 'activity-item';
    logItem.innerHTML = `
        <span class="activity-timestamp">${timestamp}</span>
        <span class="activity-message">${message}</span>
        <span class="activity-progress">${progress}%</span>
    `;
    
    activityStream.insertBefore(logItem, activityStream.firstChild);
    
    // Limit activity logs to prevent memory issues
    while (activityStream.children.length > MAX_ACTIVITY_LOGS) {
        activityStream.removeChild(activityStream.lastChild);
    }
}

function updateConnectionStatus(connected) {
    const statusIndicator = document.getElementById('activity-status');
    if (statusIndicator) {
        statusIndicator.className = connected ? 'status-connected' : 'status-disconnected';
        statusIndicator.title = connected ? 'Connected' : 'Disconnected';
    }
}

// Utility functions
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.innerHTML = `
        <span>${message}</span>
        <button style="margin-left: auto; background: none; border: none; font-size: 1.2rem; cursor: pointer;" 
                onclick="this.parentElement.remove()">×</button>
    `;
    alertDiv.style.display = 'flex';
    alertDiv.style.alignItems = 'center';
    alertDiv.style.justifyContent = 'space-between';
    alertDiv.style.padding = '10px';
    alertDiv.style.margin = '10px 0';
    alertDiv.style.borderRadius = '5px';
    alertDiv.style.border = '1px solid';
    
    // Style based on type
    switch (type) {
        case 'success':
            alertDiv.style.backgroundColor = '#d4edda';
            alertDiv.style.color = '#155724';
            alertDiv.style.borderColor = '#c3e6cb';
            break;
        case 'error':
            alertDiv.style.backgroundColor = '#f8d7da';
            alertDiv.style.color = '#721c24';
            alertDiv.style.borderColor = '#f5c6cb';
            break;
        case 'warning':
            alertDiv.style.backgroundColor = '#fff3cd';
            alertDiv.style.color = '#856404';
            alertDiv.style.borderColor = '#ffeaa7';
            break;
        default:
            alertDiv.style.backgroundColor = '#d1ecf1';
            alertDiv.style.color = '#0c5460';
            alertDiv.style.borderColor = '#bee5eb';
    }
    
    const container = document.querySelector('.container');
    const dashboardGrid = document.querySelector('.dashboard-grid');
    if (container && dashboardGrid) {
        container.insertBefore(alertDiv, dashboardGrid);
    } else {
        document.body.appendChild(alertDiv);
    }
    
    // Auto-remove after delay
    setTimeout(() => {
        if (alertDiv.parentElement) {
            alertDiv.remove();
        }
    }, ALERT_AUTO_REMOVE_DELAY);
}

function showLoading(message = 'Processing...') {
    const loadingMessage = document.getElementById('loading-message');
    const loadingOverlay = document.getElementById('loading-overlay');
    
    if (loadingMessage) loadingMessage.textContent = message;
    if (loadingOverlay) loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    const loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) loadingOverlay.style.display = 'none';
}

function showModelDetails(modelName) {
    showAlert(`Detailed view for ${modelName} - Feature coming soon!`, 'info');
}

// Error handling for async functions
function handleAsyncError(error, context = 'Operation') {
    console.error(`${context} failed:`, error);
    showAlert(`${context} failed: ${error.message}`, 'error');
    hideLoading();
}

// Initialize WebSocket connection when page loads
document.addEventListener('DOMContentLoaded', function() {
    socket.connect();
});