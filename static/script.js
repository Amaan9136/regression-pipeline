const socket = io();
let sessionId = null;
let currentDataset = null;
let cleaningOperations = {};

const ALERT_AUTO_REMOVE_DELAY = 5000;
const MAX_ACTIVITY_LOGS = 50;

document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    setupWebSocket();
});

function initializeEventListeners() {
    // File upload elements
    const fileInput = document.getElementById('file-input');
    const uploadZone = document.getElementById('upload-zone');
    const datasetSelect = document.getElementById('dataset-select');
    
    // Drag and drop functionality
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleFileDrop);
    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    
    // Dataset selection
    datasetSelect.addEventListener('change', handleDatasetSelection);
    
    // Action buttons
    document.getElementById('clean-data-btn').addEventListener('click', showCleaningInterface);
    document.getElementById('train-models-btn').addEventListener('click', startTraining);
    document.getElementById('apply-cleaning-btn').addEventListener('click', applyCleaningOperations);
    
    // Cleaning tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });
}

function setupWebSocket() {
    socket.on('session_created', (data) => {
        sessionId = data.session_id;
        addActivityLog('Session created successfully', 0);
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
    });

    socket.on('connect', () => {
        console.log('Connected to server');
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        showAlert('Connection lost. Please refresh the page.', 'warning');
    });
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
            document.getElementById('dataset-select').value = result.filename;
            currentDataset = result.filename;
            await loadDatasetInfo(result.filename);
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
    addActivityLog(`Selected dataset: ${filename}`, 100);
}

async function loadDatasetInfo(filename) {
    try {
        showLoading('Loading dataset information...');
        
        const response = await fetch(`/api/get_column_info/${filename}`);
        const result = await response.json();
        
        if (result.success) {
            populateDatasetOverview(result.column_info, result.dataset_shape);
            populateCleaningTabs(result.column_info);
            document.getElementById('clean-data-btn').disabled = false;
        } else {
            showAlert(`Failed to load dataset info: ${result.error}`, 'error');
        }
    } catch (error) {
        showAlert(`Error loading dataset: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

// Data cleaning interface functions
function showCleaningInterface() {
    if (!currentDataset) {
        showAlert('Please select a dataset first', 'warning');
        return;
    }
    
    document.getElementById('cleaning-interface').style.display = 'block';
    addActivityLog('Opened data cleaning interface', 0);
}

function switchTab(tabName) {
    // Remove active class from all tabs and content
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    // Add active class to selected tab and content
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(`${tabName}-tab`).classList.add('active');
}

function populateDatasetOverview(columnInfo, shape) {
    const overview = document.getElementById('dataset-overview');
    const totalMissing = Object.values(columnInfo).reduce((sum, col) => sum + col.missing_count, 0);
    const categoricalCount = Object.values(columnInfo).filter(col => col.dtype === 'object').length;
    
    overview.innerHTML = `
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">${shape[0].toLocaleString()}</div>
                <div class="metric-label">Rows</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${shape[1]}</div>
                <div class="metric-label">Columns</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${totalMissing.toLocaleString()}</div>
                <div class="metric-label">Missing Values</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${categoricalCount}</div>
                <div class="metric-label">Categorical Columns</div>
            </div>
        </div>
    `;
}

function populateCleaningTabs(columnInfo) {
    // Columns tab
    populateColumnsTab(columnInfo);
    
    // Missing values tab
    populateMissingValuesTab(columnInfo);
    
    // Encoding tab
    populateEncodingTab(columnInfo);
}

function populateColumnsTab(columnInfo) {
    const columnsList = document.getElementById('columns-list');
    columnsList.innerHTML = Object.entries(columnInfo).map(([colName, info]) => `
        <div class="column-item">
            <div class="column-info">
                <div class="column-name">${colName}</div>
                <div class="column-details">
                    Type: ${info.dtype} | Missing: ${info.missing_count} (${info.missing_percentage.toFixed(1)}%) | 
                    Unique: ${info.unique_values}
                </div>
            </div>
            <div class="column-actions">
                <input type="text" class="input-field" placeholder="Rename to..." 
                       onchange="updateRename('${colName}', this.value)">
                <button class="btn btn-warning" onclick="markForDrop('${colName}')">Drop</button>
            </div>
        </div>
    `).join('');
}

function populateMissingValuesTab(columnInfo) {
    const missingContent = document.getElementById('missing-values-content');
    const columnsWithMissing = Object.entries(columnInfo).filter(([_, info]) => info.missing_count > 0);
    
    if (columnsWithMissing.length === 0) {
        missingContent.innerHTML = '<div class="alert alert-success">✅ No missing values detected!</div>';
        return;
    }

    missingContent.innerHTML = `
        <div class="alert alert-info">📊 ${columnsWithMissing.length} columns have missing values</div>
        <div class="column-list">
            ${columnsWithMissing.map(([colName, info]) => `
                <div class="column-item">
                    <div class="column-info">
                        <div class="column-name">${colName}</div>
                        <div class="column-details">
                            Missing: ${info.missing_count} values (${info.missing_percentage.toFixed(1)}%)
                        </div>
                    </div>
                    <div class="column-actions">
                        <select class="select-field" onchange="updateMissingStrategy('${colName}', this.value)">
                            <option value="">Choose strategy...</option>
                            <option value="drop">Drop rows</option>
                            ${info.dtype === 'object' ? '' : '<option value="mean">Fill with mean</option>'}
                            ${info.dtype === 'object' ? '' : '<option value="median">Fill with median</option>'}
                            <option value="mode">Fill with mode</option>
                            <option value="forward_fill">Forward fill</option>
                            <option value="backward_fill">Backward fill</option>
                            <option value="custom_value">Custom value</option>
                        </select>
                        <input type="text" class="input-field" id="custom-${colName}" 
                               placeholder="Enter custom value" style="display: none;">
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

function populateEncodingTab(columnInfo) {
    const encodingContent = document.getElementById('encoding-content');
    const categoricalColumns = Object.entries(columnInfo).filter(([_, info]) => info.dtype === 'object');
    
    if (categoricalColumns.length === 0) {
        encodingContent.innerHTML = '<div class="alert alert-info">No categorical columns found</div>';
        return;
    }

    encodingContent.innerHTML = `
        <div class="alert alert-info">🔤 ${categoricalColumns.length} categorical columns found</div>
        <div class="column-list">
            ${categoricalColumns.map(([colName, info]) => `
                <div class="column-item">
                    <div class="column-info">
                        <div class="column-name">${colName}</div>
                        <div class="column-details">
                            Unique values: ${info.unique_values} | 
                            Sample: ${info.sample_values.slice(0, 3).join(', ')}...
                        </div>
                    </div>
                    <div class="column-actions">
                        <select class="select-field" onchange="updateEncodingStrategy('${colName}', this.value)">
                            <option value="">Choose encoding...</option>
                            <option value="label">Label Encoding</option>
                            <option value="onehot">One-Hot Encoding</option>
                            <option value="ordinal">Ordinal Encoding</option>
                        </select>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

// Cleaning operation handlers
function updateRename(columnName, newName) {
    if (!cleaningOperations.rename_columns) cleaningOperations.rename_columns = {};
    if (newName.trim()) {
        cleaningOperations.rename_columns[columnName] = newName.trim();
        addActivityLog(`Marked "${columnName}" for rename to "${newName.trim()}"`, 0);
    } else {
        delete cleaningOperations.rename_columns[columnName];
    }
}

function markForDrop(columnName) {
    if (!cleaningOperations.drop_columns) cleaningOperations.drop_columns = [];
    if (!cleaningOperations.drop_columns.includes(columnName)) {
        cleaningOperations.drop_columns.push(columnName);
        showAlert(`Marked "${columnName}" for deletion`, 'info');
        addActivityLog(`Marked "${columnName}" for deletion`, 0);
    }
}

function updateMissingStrategy(columnName, strategy) {
    if (!cleaningOperations.missing_values) cleaningOperations.missing_values = {};
    
    cleaningOperations.missing_values[columnName] = { method: strategy };
    
    // Show/hide custom value input
    const customInput = document.getElementById(`custom-${columnName}`);
    if (strategy === 'custom_value' && customInput) {
        customInput.style.display = 'block';
        customInput.onchange = (e) => {
            cleaningOperations.missing_values[columnName].value = e.target.value;
        };
    } else if (customInput) {
        customInput.style.display = 'none';
    }
    
    addActivityLog(`Set missing value strategy for "${columnName}": ${strategy}`, 0);
}

function updateEncodingStrategy(columnName, strategy) {
    if (!cleaningOperations.categorical_encoding) cleaningOperations.categorical_encoding = {};
    cleaningOperations.categorical_encoding[columnName] = { method: strategy };
    addActivityLog(`Set encoding strategy for "${columnName}": ${strategy}`, 0);
}

async function applyCleaningOperations() {
    if (!currentDataset || Object.keys(cleaningOperations).length === 0) {
        showAlert('No cleaning operations to apply', 'warning');
        return;
    }

    try {
        showLoading('Applying data cleaning operations...');
        addActivityLog('Applying cleaning operations...', 0);
        
        const response = await fetch('/api/apply_cleaning', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: currentDataset,
                operations: cleaningOperations
            })
        });

        const result = await response.json();
        
        if (result.success) {
            showAlert(`Data cleaned successfully! New file: ${result.cleaned_filename}`, 'success');
            addDatasetToSelect(result.cleaned_filename);
            document.getElementById('dataset-select').value = result.cleaned_filename;
            currentDataset = result.cleaned_filename;
            addActivityLog(`Data cleaning completed`, 100);
            
            // Reset cleaning operations
            cleaningOperations = {};
            
            // Display cleaning summary
            displayCleaningSummary(result);
        } else {
            showAlert('Cleaning failed: ' + result.error, 'error');
        }
    } catch (error) {
        showAlert('Cleaning error: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

function displayCleaningSummary(result) {
    const summaryHtml = `
        <div class="alert alert-success">
            <h4>🎉 Cleaning Summary</h4>
            <p><strong>Original shape:</strong> ${result.original_shape[0]} × ${result.original_shape[1]}</p>
            <p><strong>Final shape:</strong> ${result.final_shape[0]} × ${result.final_shape[1]}</p>
            <p><strong>Quality Score:</strong> ${result.quality_report.quality_score.toFixed(1)}/100</p>
            ${result.validation.is_valid ? 
                '<p>✅ Dataset is ready for training!</p>' : 
                '<p>⚠️ Dataset may need additional cleaning</p>'
            }
        </div>
    `;
    
    const overviewTab = document.getElementById('overview-tab');
    overviewTab.innerHTML = summaryHtml + overviewTab.innerHTML;
}

// Training functions
async function startTraining() {
    if (!currentDataset || !sessionId) {
        showAlert('Please select a dataset and ensure connection is established', 'warning');
        return;
    }

    // Get training configuration
    const config = {
        test_size: 0.2,
        cross_validation_folds: 5,
        hyperparameter_tuning: true,
        feature_engineering: true
    };

    addActivityLog('Starting model training...', 0);
    document.getElementById('results-section').style.display = 'none';
    
    // Emit training start event
    socket.emit('start_training', {
        session_id: sessionId,
        dataset: currentDataset,
        config: config
    });
}

function displayTrainingResults(data) {
    document.getElementById('results-section').style.display = 'block';
    
    // Update model comparison table
    updateModelComparison(data.models);
    
    // Update visualizations
    if (data.plots) {
        updateVisualizationGrid(data.plots);
    }
    
    addActivityLog('Model training completed successfully!', 100);
    hideLoading();
}

function updateModelComparison(models) {
    const tableBody = document.getElementById('model-comparison-body');
    
    // Find best model (highest R²)
    const bestModel = Object.keys(models).reduce((best, current) => 
        models[current].r2 > models[best].r2 ? current : best
    );
    
    tableBody.innerHTML = Object.entries(models).map(([modelName, results]) => `
        <tr class="${modelName === bestModel ? 'best-model-row' : ''}">
            <td><strong>${modelName}</strong></td>
            <td>${results.r2.toFixed(4)}</td>
            <td>${results.mse.toFixed(4)}</td>
            <td>${results.mae.toFixed(4)}</td>
            <td>${results.rmse.toFixed(4)}</td>
            <td>${results.cv_mean.toFixed(3)} ± ${results.cv_std.toFixed(3)}</td>
            <td>
                <button class="btn btn-primary" onclick="showModelDetails('${modelName}')">
                    📊 Details
                </button>
            </td>
        </tr>
    `).join('');
}

function updateVisualizationGrid(plots) {
    const vizGrid = document.getElementById('visualization-grid');
    const plotTitles = {
        'comparison': '📊 Model Performance Comparison',
        'best_scatter': '🎯 Best Model: Predicted vs Actual',
        'heatmap': '🔥 Performance Metrics Heatmap',
        'residuals': '📈 Residual Analysis',
        'feature_importance': '🔑 Feature Importance'
    };

    vizGrid.innerHTML = Object.entries(plots).map(([plotType, plotPath]) => `
        <div class="viz-card">
            <div class="viz-title">${plotTitles[plotType] || 'Analysis Plot'}</div>
            <img src="/${plotPath}" alt="${plotTitles[plotType]}" 
                 onerror="this.style.display='none'" loading="lazy">
        </div>
    `).join('');
}

// Progress and activity functions
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
    
    const logItem = document.createElement('div');
    logItem.className = 'activity-item';
    logItem.innerHTML = `
        <div class="activity-content">
            <span class="activity-message">${message}</span>
            <span class="activity-progress">${progress}%</span>
        </div>
    `;
    
    activityStream.insertBefore(logItem, activityStream.firstChild);
    
    // Limit activity logs to prevent memory issues
    while (activityStream.children.length > MAX_ACTIVITY_LOGS) {
        activityStream.removeChild(activityStream.lastChild);
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
    
    const container = document.querySelector('.container');
    const dashboardGrid = document.querySelector('.dashboard-grid');
    container.insertBefore(alertDiv, dashboardGrid);
    
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
    // This would show detailed model information in a modal
    showAlert(`Detailed view for ${modelName} - Feature coming soon!`, 'info');
}

// Error handling for async functions
function handleAsyncError(error, context = 'Operation') {
    console.error(`${context} failed:`, error);
    showAlert(`${context} failed: ${error.message}`, 'error');
    hideLoading();
}

// Initialize WebSocket connection
socket.connect();