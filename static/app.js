let selectedClasses = new Set();
let currentConfidence = 0.3;
let availableCameras = [];
let trackingEnabled = false;

// Initialize with all classes selected
function initializeClasses() {
    const checkboxes = document.querySelectorAll('.class-checkbox input');
    checkboxes.forEach(checkbox => {
        selectedClasses.add(parseInt(checkbox.value));
    });
}

async function detectCameras() {
    const statusDiv = document.getElementById('cameraStatus');
    statusDiv.innerHTML = '<span class="loading">Stopping active streams...</span>';
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
                let statusMessage = `<span style="color: orange;">No working cameras found. Detected ${availableCameras.length} device(s):</span><br>`;
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
            statusDiv.innerHTML = `<span style="color: red;">Error detecting cameras: ${result.message}</span>`;
        }
    } catch (error) {
        statusDiv.innerHTML = `<span style="color: red;">Error: ${error.message}</span>`;
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
    const timestamp = new Date().getTime();
    videoElement.src = `/video_feed/${encodeURIComponent(cameraSource)}?t=${timestamp}`;
    videoElement.style.display = 'block';
    console.log('Started YOLO stream:', videoElement.src);
}

function startTestStream() {
    const cameraSource = document.getElementById('cameraSource').value;
    const videoElement = document.getElementById('videoStream');
    const timestamp = new Date().getTime();
    videoElement.src = `/test_video_feed/${encodeURIComponent(cameraSource)}?t=${timestamp}`;
    videoElement.style.display = 'block';
    console.log('Started test stream:', videoElement.src);
}

function stopStream() {
    const videoElement = document.getElementById('videoStream');
    videoElement.src = '';
    videoElement.style.display = 'none';
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
            const videoElement = document.getElementById('videoStream');
            if (videoElement.src && videoElement.style.display !== 'none') {
                setTimeout(() => {
                    startStream();
                }, 500);
            }
        } else {
            statusDiv.innerHTML = `<span style="color: red;">Error: ${result.detail}</span>`;
        }
    } catch (error) {
        statusDiv.innerHTML = `<span style="color: red;">Error loading model: ${error.message}</span>`;
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
            gpuStatusDiv.innerHTML = '<span style="color: green;">GPU memory cleared successfully</span>';
            setTimeout(updateGpuStatus, 1000);
        } else {
            gpuStatusDiv.innerHTML = `<span style="color: red;">Error: ${result.message}</span>`;
        }
    } catch (error) {
        gpuStatusDiv.innerHTML = `<span style="color: red;">Error clearing memory: ${error.message}</span>`;
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
        } else {
            gpuStatusDiv.innerHTML = 'GPU: Not available (CPU mode)';
        }
    } catch (error) {
        document.getElementById('gpuStatus').innerHTML = 'GPU status: Error loading';
    }
}

function updateClassCheckboxes(classNames) {
    const container = document.getElementById('classCheckboxes');
    let checkboxesHtml = '';
    selectedClasses.clear();
    Object.entries(classNames).forEach(([classId, className]) => {
        checkboxesHtml += `
            <label class="class-checkbox">
                <input type="checkbox" value="${classId}" checked onchange="updateClassFilter()"> ${className}
            </label>
        `;
        selectedClasses.add(parseInt(classId));
    });
    container.innerHTML = checkboxesHtml;
    sendSettings();
}

function updateTracking() {
    const toggle = document.getElementById('trackingToggle');
    const status = document.getElementById('trackingStatus');
    trackingEnabled = toggle.checked;
    status.textContent = trackingEnabled ? 'On' : 'Off';
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
            tracking: trackingEnabled
        })
    });
}

async function uploadModel() {
    const fileInput = document.getElementById('modelUpload');
    const uploadStatus = document.getElementById('uploadStatus');
    if (!fileInput.files.length) {
        uploadStatus.innerHTML = '<span style="color: red;">Please select a .pt file to upload.</span>';
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
            uploadStatus.innerHTML = `<span style="color: green;">${result.message}</span>`;
            // Refresh model selector and select the new model
            await refreshModelSelector(result.model_path);
        } else {
            uploadStatus.innerHTML = `<span style="color: red;">${result.message}</span>`;
        }
    } catch (error) {
        uploadStatus.innerHTML = `<span style="color: red;">Error: ${error.message}</span>`;
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

window.onload = function() {
    initializeClasses();
    updateGpuStatus();
    detectCameras();
    document.getElementById('trackingToggle').checked = false;
    document.getElementById('trackingStatus').textContent = 'Off';
}; 