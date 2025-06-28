// Analytics and Metrics Variables
let selectedClasses = new Set();
let currentConfidence = 0.3;
let availableCameras = [];
let trackingEnabled = false;
let trackThresh = 0.5;
let trackBuffer = 30;
let matchThresh = 0.8;

// Analytics Data
let analyticsData = {
    totalDetections: 0,
    fps: 0,
    activeObjects: 0,
    confidenceAvg: 0,
    detectionHistory: [],
    classDistribution: {},
    performanceMetrics: {
        avgProcessingTime: 0,
        memoryUsage: 0,
        gpuUtilization: 0
    },
    systemStatus: {
        gpu: 'offline',
        model: 'offline',
        stream: 'offline',
        memory: 'offline'
    }
};

// Chart instances
let detectionChart = null;
let classChart = null;

// Initialize with all classes selected
function initializeClasses() {
    const checkboxes = document.querySelectorAll('.class-checkbox input');
    checkboxes.forEach(checkbox => {
        selectedClasses.add(parseInt(checkbox.value));
    });
}

// Initialize Analytics Dashboard
function initializeAnalytics() {
    initializeCharts();
    startMetricsUpdate();
    updateSystemStatus();
}

// Initialize Charts
function initializeCharts() {
    // Detection Rate Chart
    const detectionCtx = document.getElementById('detectionChart');
    if (detectionCtx) {
        detectionChart = new Chart(detectionCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Detections per Second',
                    data: [],
                    borderColor: '#ffffff',
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#888' },
                        grid: { color: '#333' }
                    },
                    y: {
                        ticks: { color: '#888' },
                        grid: { color: '#333' }
                    }
                }
            }
        });
    }

    // Class Distribution Chart
    const classCtx = document.getElementById('classChart');
    if (classCtx) {
        classChart = new Chart(classCtx, {
            type: 'doughnut',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        '#ffffff', '#cccccc', '#999999', '#666666',
                        '#ff6666', '#66ff66', '#6666ff', '#ffff66'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#ffffff',
                            padding: 10
                        }
                    }
                }
            }
        });
    }
}

// Update Analytics Metrics
function updateAnalyticsMetrics() {
    // Update metric cards
    document.getElementById('totalDetections').textContent = analyticsData.totalDetections.toLocaleString();
    document.getElementById('fpsCounter').textContent = analyticsData.fps.toFixed(1);
    document.getElementById('activeObjects').textContent = analyticsData.activeObjects;
    document.getElementById('confidenceAvg').textContent = analyticsData.confidenceAvg.toFixed(2);

    // Update trends
    updateTrends();
    
    // Update charts
    updateCharts();
    
    // Update detection history
    updateDetectionHistory();
}

// Update Trends
function updateTrends() {
    const trends = calculateTrends();
    
    document.getElementById('detectionTrend').textContent = trends.detection;
    document.getElementById('fpsTrend').textContent = trends.fps;
    document.getElementById('objectsTrend').textContent = trends.objects;
    document.getElementById('confidenceTrend').textContent = trends.confidence;
    
    // Update trend colors
    Object.keys(trends).forEach(key => {
        const element = document.getElementById(`${key}Trend`);
        if (element) {
            element.className = `metric-trend ${trends[key].includes('+') ? 'positive' : 'negative'}`;
        }
    });
}

// Calculate Trends
function calculateTrends() {
    const recent = analyticsData.detectionHistory.slice(-10);
    const previous = analyticsData.detectionHistory.slice(-20, -10);
    
    const recentAvg = recent.length > 0 ? recent.reduce((sum, d) => sum + d.count, 0) / recent.length : 0;
    const previousAvg = previous.length > 0 ? previous.reduce((sum, d) => sum + d.count, 0) / previous.length : 0;
    
    const change = previousAvg > 0 ? ((recentAvg - previousAvg) / previousAvg * 100).toFixed(1) : 0;
    
    return {
        detection: `${change >= 0 ? '+' : ''}${change}% vs last 10 detections`,
        fps: 'Real-time',
        objects: `${analyticsData.activeObjects} currently tracking`,
        confidence: 'Last 100 detections'
    };
}

// Update Charts
function updateCharts() {
    if (detectionChart) {
        const recentData = analyticsData.detectionHistory.slice(-20);
        detectionChart.data.labels = recentData.map(d => d.time);
        detectionChart.data.datasets[0].data = recentData.map(d => d.count);
        detectionChart.update('none');
    }
    
    if (classChart) {
        const classData = Object.entries(analyticsData.classDistribution);
        classChart.data.labels = classData.map(([name, count]) => name);
        classChart.data.datasets[0].data = classData.map(([name, count]) => count);
        classChart.update('none');
    }
}

// Update Detection History Table
function updateDetectionHistory() {
    const tbody = document.getElementById('detectionTableBody');
    if (!tbody) return;
    
    const recentDetections = analyticsData.detectionHistory.slice(-10).reverse();
    
    tbody.innerHTML = recentDetections.map(detection => `
        <tr>
            <td>${detection.time}</td>
            <td>${detection.class}</td>
            <td>${(detection.confidence * 100).toFixed(1)}%</td>
            <td><span class="detection-count">${detection.count}</span></td>
            <td>${detection.status}</td>
        </tr>
    `).join('');
}

// Update System Status
function updateSystemStatus() {
    // GPU Status
    const gpuIndicator = document.getElementById('gpuStatus');
    const gpuInfo = document.getElementById('gpuInfo');
    if (gpuIndicator && gpuInfo) {
        gpuIndicator.className = `status-indicator ${analyticsData.systemStatus.gpu}`;
        gpuInfo.textContent = analyticsData.systemStatus.gpu === 'online' ? 'Active' : 'Inactive';
    }
    
    // Model Status
    const modelIndicator = document.getElementById('modelStatus');
    const modelInfo = document.getElementById('modelInfo');
    if (modelIndicator && modelInfo) {
        modelIndicator.className = `status-indicator ${analyticsData.systemStatus.model}`;
        modelInfo.textContent = analyticsData.systemStatus.model === 'online' ? 'Loaded' : 'Not Loaded';
    }
    
    // Stream Status
    const streamIndicator = document.getElementById('streamStatus');
    const streamInfo = document.getElementById('streamInfo');
    if (streamIndicator && streamInfo) {
        streamIndicator.className = `status-indicator ${analyticsData.systemStatus.stream}`;
        streamInfo.textContent = analyticsData.systemStatus.stream === 'online' ? 'Live' : 'Idle';
    }
    
    // Memory Status
    const memoryIndicator = document.getElementById('memoryStatus');
    const memoryInfo = document.getElementById('memoryInfo');
    if (memoryIndicator && memoryInfo) {
        memoryIndicator.className = `status-indicator ${analyticsData.systemStatus.memory}`;
        memoryInfo.textContent = analyticsData.systemStatus.memory === 'online' ? 'Optimal' : 'High Usage';
    }
}

// Start Metrics Update Loop
function startMetricsUpdate() {
    setInterval(() => {
        // Simulate real-time data updates
        if (analyticsData.systemStatus.stream === 'online') {
            // Simulate detection data
            const newDetection = {
                time: new Date().toLocaleTimeString(),
                class: getRandomClass(),
                confidence: Math.random() * 0.5 + 0.5,
                count: Math.floor(Math.random() * 5) + 1,
                status: 'Active'
            };
            
            analyticsData.detectionHistory.push(newDetection);
            analyticsData.totalDetections += newDetection.count;
            analyticsData.fps = Math.random() * 10 + 20;
            analyticsData.activeObjects = Math.floor(Math.random() * 10) + 1;
            analyticsData.confidenceAvg = Math.random() * 0.3 + 0.7;
            
            // Update class distribution
            if (!analyticsData.classDistribution[newDetection.class]) {
                analyticsData.classDistribution[newDetection.class] = 0;
            }
            analyticsData.classDistribution[newDetection.class] += newDetection.count;
            
            // Keep only last 100 detections
            if (analyticsData.detectionHistory.length > 100) {
                analyticsData.detectionHistory.shift();
            }
            
            updateAnalyticsMetrics();
        }
    }, 2000);
}

// Utility function to get random class
function getRandomClass() {
    const classes = ['person', 'car', 'dog', 'cat', 'bicycle', 'motorcycle', 'bus', 'truck'];
    return classes[Math.floor(Math.random() * classes.length)];
}

// Clear Detection History
function clearHistory() {
    analyticsData.detectionHistory = [];
    analyticsData.totalDetections = 0;
    analyticsData.classDistribution = {};
    updateAnalyticsMetrics();
}

// Export Detection History
function exportHistory() {
    const dataStr = JSON.stringify(analyticsData.detectionHistory, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `detection_history_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
}

// Enhanced Camera Detection
async function detectCameras() {
    const statusDiv = document.getElementById('cameraStatus');
    const detectBtn = document.querySelector('button[onclick="detectCameras()"]');
    const spinner = detectBtn.querySelector('.loading-spinner');
    
    statusDiv.innerHTML = '<span class="loading">Stopping active streams...</span>';
    spinner.style.display = 'inline-block';
    detectBtn.disabled = true;
    
    stopStream();
    await new Promise(resolve => setTimeout(resolve, 1000));
    statusDiv.innerHTML = '<span class="loading">Detecting cameras...</span>';
    
    try {
        const response = await fetch('/get_cameras');
        const result = await response.json();
        if (result.status === 'success') {
            availableCameras = result.cameras;
            updateCameraSelector();
            const workingCameras = availableCameras.filter(cam => cam.status === 'working');
            if (workingCameras.length > 0) {
                statusDiv.innerHTML = `Found ${workingCameras.length} working camera(s) out of ${availableCameras.length} detected`;
                document.getElementById('cameraSource').value = workingCameras[0].index;
            } else {
                let statusMessage = `<span style="color: #ffaa00;">No working cameras found. Detected ${availableCameras.length} device(s):</span><br>`;
                availableCameras.forEach(cam => {
                    const statusIcon = cam.status === 'working' ? '✅' : 
                                     cam.status === 'no_signal' ? '⚠️' : 
                                     cam.status === 'cannot_open' ? '🔒' : '❌';
                    statusMessage += `${statusIcon} Camera ${cam.index}: ${cam.status} ${cam.error ? '(' + cam.error + ')' : ''}<br>`;
                });
                statusMessage += '<br><strong>Try:</strong> Stop the video stream first, then detect cameras. Camera 1 is known to work on your system.';
                statusDiv.innerHTML = statusMessage;
            }
        } else {
            statusDiv.innerHTML = `<span style="color: #ff6666;">Error detecting cameras: ${result.message}</span>`;
        }
    } catch (error) {
        statusDiv.innerHTML = `<span style="color: #ff6666;">Error: ${error.message}</span>`;
    } finally {
        spinner.style.display = 'none';
        detectBtn.disabled = false;
    }
}

function updateCameraSelector() {
    const selector = document.getElementById('cameraSelector');
    selector.innerHTML = '<option value="">Select Camera...</option>';
    availableCameras.forEach(camera => {
        const option = document.createElement('option');
        option.value = camera.index;
        const statusIcon = camera.status === 'working' ? '✅' : 
                         camera.status === 'no_signal' ? '⚠️' : 
                         camera.status === 'cannot_open' ? '🔒' : '❌';
        if (camera.status === 'working') {
            option.textContent = `${statusIcon} ${camera.name} (${camera.resolution}, ${camera.fps}fps)`;
        } else {
            option.textContent = `${statusIcon} ${camera.name} - ${camera.status}`;
            option.disabled = true;
            option.style.color = '#999';
        }
        selector.appendChild(option);
    });
    const rtspOption = document.createElement('option');
    rtspOption.value = 'rtsp';
    rtspOption.textContent = '📡 Enter RTSP URL manually';
    selector.appendChild(rtspOption);
}

function selectCamera() {
    const selector = document.getElementById('cameraSelector');
    const cameraSource = document.getElementById('cameraSource');
    if (selector.value) {
        cameraSource.value = selector.value;
        const camera = availableCameras.find(cam => cam.index == selector.value);
        if (camera) {
            document.getElementById('cameraStatus').innerHTML = 
                `Selected: ${camera.name} (${camera.resolution}, ${camera.fps}fps)`;
        }
    }
}

function startStream() {
    const cameraSource = document.getElementById('cameraSource').value;
    const videoElement = document.getElementById('videoStream');
    const placeholder = document.getElementById('videoPlaceholder');
    const timestamp = new Date().getTime();
    
    videoElement.src = `/video_feed/${encodeURIComponent(cameraSource)}?t=${timestamp}`;
    videoElement.style.display = 'block';
    placeholder.style.display = 'none';
    
    // Update system status
    analyticsData.systemStatus.stream = 'online';
    updateSystemStatus();
    
    console.log('Started YOLO stream:', videoElement.src);
}

function startTestStream() {
    const cameraSource = document.getElementById('cameraSource').value;
    const videoElement = document.getElementById('videoStream');
    const placeholder = document.getElementById('videoPlaceholder');
    const timestamp = new Date().getTime();
    
    videoElement.src = `/test_video_feed/${encodeURIComponent(cameraSource)}?t=${timestamp}`;
    videoElement.style.display = 'block';
    placeholder.style.display = 'none';
    
    // Update system status
    analyticsData.systemStatus.stream = 'online';
    updateSystemStatus();
    
    console.log('Started test stream:', videoElement.src);
}

function stopStream() {
    const videoElement = document.getElementById('videoStream');
    const placeholder = document.getElementById('videoPlaceholder');
    videoElement.src = '';
    videoElement.style.display = 'none';
    placeholder.style.display = 'block';
    
    // Update system status
    analyticsData.systemStatus.stream = 'offline';
    updateSystemStatus();
}

function updateConfidence(value) {
    currentConfidence = parseFloat(value);
    document.getElementById('confidenceValue').textContent = value;
    sendSettings();
}

function updateClassFilter() {
    selectedClasses.clear();
    const checkboxes = document.querySelectorAll('.class-checkbox input:checked');
    checkboxes.forEach(checkbox => {
        selectedClasses.add(parseInt(checkbox.value));
    });
    sendSettings();
}

function selectAllClasses() {
    const checkboxes = document.querySelectorAll('.class-checkbox input');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
    });
    updateClassFilter();
}

function deselectAllClasses() {
    const checkboxes = document.querySelectorAll('.class-checkbox input');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
    updateClassFilter();
}

async function loadModel() {
    const modelSelector = document.getElementById('modelSelector');
    const modelPath = modelSelector.value;
    const statusDiv = document.getElementById('modelStatus');
    statusDiv.innerHTML = '<span class="loading">Loading model...</span>';
    
    // Update system status
    analyticsData.systemStatus.model = 'warning';
    updateSystemStatus();
    
    try {
        const response = await fetch('/load_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_path: modelPath
            })
        });
        const result = await response.json();
        if (result.status === 'success') {
            statusDiv.innerHTML = `Current: ${result.model_path} (${result.num_classes} classes)`;
            updateClassCheckboxes(result.class_names);
            updateGpuStatus();
            
            // Update system status
            analyticsData.systemStatus.model = 'online';
            updateSystemStatus();
            
            const videoElement = document.getElementById('videoStream');
            if (videoElement.src && videoElement.style.display !== 'none') {
                setTimeout(() => {
                    startStream();
                }, 500);
            }
        } else {
            statusDiv.innerHTML = `<span class="error">Error: ${result.detail}</span>`;
            analyticsData.systemStatus.model = 'offline';
            updateSystemStatus();
        }
    } catch (error) {
        statusDiv.innerHTML = `<span class="error">Error loading model: ${error.message}</span>`;
        analyticsData.systemStatus.model = 'offline';
        updateSystemStatus();
    }
}

async function clearGpuMemory() {
    const gpuStatusDiv = document.getElementById('gpuStatus');
    gpuStatusDiv.innerHTML = '<span class="loading">Clearing GPU memory...</span>';
    try {
        const response = await fetch('/clear_gpu_memory', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        const result = await response.json();
        if (result.status === 'success') {
            gpuStatusDiv.innerHTML = '<span class="success">GPU memory cleared successfully</span>';
            setTimeout(updateGpuStatus, 1000);
        } else {
            gpuStatusDiv.innerHTML = `<span class="error">Error: ${result.message}</span>`;
        }
    } catch (error) {
        gpuStatusDiv.innerHTML = `<span class="error">Error clearing memory: ${error.message}</span>`;
    }
}

async function updateGpuStatus() {
    try {
        const response = await fetch('/gpu_status');
        const gpuInfo = await response.json();
        const gpuStatusDiv = document.getElementById('gpuStatus');
        if (gpuInfo.gpu_available) {
            const allocatedPercent = ((gpuInfo.allocated_memory_gb / gpuInfo.total_memory_gb) * 100).toFixed(1);
            gpuStatusDiv.innerHTML = `
                GPU: ${gpuInfo.device_name}<br>
                Memory: ${gpuInfo.allocated_memory_gb}GB / ${gpuInfo.total_memory_gb}GB (${allocatedPercent}%)
            `;
            
            // Update system status
            analyticsData.systemStatus.gpu = 'online';
            analyticsData.systemStatus.memory = allocatedPercent > 80 ? 'warning' : 'online';
            updateSystemStatus();
        } else {
            gpuStatusDiv.innerHTML = 'GPU: Not available (CPU mode)';
            analyticsData.systemStatus.gpu = 'offline';
            updateSystemStatus();
        }
    } catch (error) {
        document.getElementById('gpuStatus').innerHTML = 'GPU status: Error loading';
        analyticsData.systemStatus.gpu = 'offline';
        updateSystemStatus();
    }
}

function updateClassCheckboxes(classNames) {
    const container = document.getElementById('classCheckboxes');
    let checkboxesHtml = '';
    selectedClasses.clear();
    Object.entries(classNames).forEach(([classId, className]) => {
        checkboxesHtml += `
            <div class="class-checkbox">
                <input type="checkbox" id="class_${classId}" value="${classId}" checked onchange="updateClassFilter()">
                <label for="class_${classId}">${className}</label>
            </div>
        `;
        selectedClasses.add(parseInt(classId));
    });
    container.innerHTML = checkboxesHtml;
    sendSettings();
}

function toggleTracking() {
    const toggle = document.getElementById('trackingToggle');
    const status = document.getElementById('trackingStatus');
    const paramsDiv = document.getElementById('trackingParams');
    
    trackingEnabled = !trackingEnabled;
    
    if (trackingEnabled) {
        toggle.classList.add('active');
        status.textContent = 'On';
        paramsDiv.classList.add('active');
    } else {
        toggle.classList.remove('active');
        status.textContent = 'Off';
        paramsDiv.classList.remove('active');
    }
    
    sendSettings();
}

function updateTrackThresh(value) {
    trackThresh = parseFloat(value);
    document.getElementById('trackThreshValue').textContent = value;
    sendSettings();
}

function updateTrackBuffer(value) {
    trackBuffer = parseInt(value);
    document.getElementById('trackBufferValue').textContent = value;
    sendSettings();
}

function updateMatchThresh(value) {
    matchThresh = parseFloat(value);
    document.getElementById('matchThreshValue').textContent = value;
    sendSettings();
}

function sendSettings() {
    fetch('/update_settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            selected_classes: Array.from(selectedClasses),
            confidence: currentConfidence,
            tracking: trackingEnabled,
            track_thresh: trackThresh,
            track_buffer: trackBuffer,
            match_thresh: matchThresh
        })
    });
}

async function uploadModel() {
    const fileInput = document.getElementById('modelUpload');
    const uploadStatus = document.getElementById('uploadStatus');
    if (!fileInput.files.length) {
        uploadStatus.innerHTML = '<span class="error">Please select a .pt file to upload.</span>';
        return;
    }
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    uploadStatus.innerHTML = '<span class="loading">Uploading model...</span>';
    try {
        const response = await fetch('/upload_model', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        if (result.status === 'success') {
            uploadStatus.innerHTML = `<span class="success">${result.message}</span>`;
            // Refresh model selector and select the new model
            await refreshModelSelector(result.model_path);
        } else {
            uploadStatus.innerHTML = `<span class="error">${result.message}</span>`;
        }
    } catch (error) {
        uploadStatus.innerHTML = `<span class="error">Error: ${error.message}</span>`;
    }
}

async function refreshModelSelector(selectModelPath) {
    // Fetch model info and update selector
    const response = await fetch('/get_model_info');
    const info = await response.json();
    const selector = document.getElementById('modelSelector');
    selector.innerHTML = '';
    info.available_models.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        if (model === selectModelPath) option.selected = true;
        selector.appendChild(option);
    });
    // Load the new model
    await loadModel();
}

// Initialize everything when page loads
window.onload = function() {
    initializeClasses();
    initializeAnalytics();
    updateGpuStatus();
    detectCameras();
    
    // Initialize tracking toggle state
    const toggle = document.getElementById('trackingToggle');
    const status = document.getElementById('trackingStatus');
    const paramsDiv = document.getElementById('trackingParams');
    
    trackingEnabled = false;
    toggle.classList.remove('active');
    status.textContent = 'Off';
    paramsDiv.classList.remove('active');
}; 