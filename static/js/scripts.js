// Initialize variables
let lossChart = null;
let startTime = new Date();
let trainingTimer = null;
let totalEpochs = 0; // Store total epochs

// Establish SocketIO connection
const socket = io();

// Listen for model info updates
socket.on('model_info', function(data) {
    document.getElementById('crypto-name').textContent = data.crypto;
    document.getElementById('model-type').textContent = data.model;
    document.getElementById('epoch-progress').textContent = `0/${data.total_epochs}`;

    // Store total epochs for future use
    totalEpochs = data.total_epochs;

    // Start timer
    startTimer();
});

// Listen for training progress updates
socket.on('update', function(data) {
    // Update progress bar
    const progressBar = document.getElementById('training-progress');
    progressBar.style.width = `${data.progress}%`;
    progressBar.textContent = `${data.progress}%`;
    progressBar.setAttribute('aria-valuenow', data.progress);

    // Update epoch display
    document.getElementById('epoch-progress').textContent =
        `${data.current_epoch}/${totalEpochs}`; // Use stored total epochs

    // Update chart data - fix epoch duplication issue
    if (lossChart) {
        // Check if data point for this epoch already exists
        const existingIndex = lossChart.data.labels.indexOf(data.current_epoch);

        if (existingIndex !== -1) {
            // Update existing data point
            lossChart.data.datasets[0].data[existingIndex] = data.loss;
            lossChart.data.datasets[1].data[existingIndex] = data.val_loss;
        } else {
            // Add new data point
            lossChart.data.labels.push(data.current_epoch);
            lossChart.data.datasets[0].data.push(data.loss);
            lossChart.data.datasets[1].data.push(data.val_loss);
        }
        lossChart.update();
    }
});

// Listen for metric updates - add model name column
socket.on('metric', function(data) {
    const tableBody = document.getElementById('metrics-table');
    const now = new Date();
    const timeString = now.toTimeString().substring(0, 8);

    const newRow = document.createElement('tr');
    newRow.innerHTML = `
        <td>${data.model_name}</td>
        <td>${data.name}</td>
        <td>${data.value.toFixed(6)}</td>
        <td>${timeString}</td>
    `;

    tableBody.appendChild(newRow);

    // Auto-scroll to bottom
    tableBody.parentElement.scrollTop = tableBody.parentElement.scrollHeight;
});

// Listen for log updates
socket.on('log', function(data) {
    const logContainer = document.getElementById('log-container');
    const now = new Date();
    const timeString = now.toTimeString().substring(0, 8);

    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    logEntry.innerHTML = `
        <span class="log-timestamp">[${timeString}]</span>
        <span class="log-message">${data.message}</span>
    `;

    logContainer.appendChild(logEntry);
    logContainer.scrollTop = logContainer.scrollHeight;
});

// Initialize chart
function initChart() {
    const ctx = document.getElementById('loss-chart').getContext('2d');
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Training Loss',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    tension: 0.1,
                    fill: true
                },
                {
                    label: 'Validation Loss',
                    data: [],
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    tension: 0.1,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Loss Value'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Training Epochs'
                    }
                }
            }
        }
    });
}

// Start timer
function startTimer() {
    clearInterval(trainingTimer);
    startTime = new Date();

    trainingTimer = setInterval(() => {
        const now = new Date();
        const diff = Math.floor((now - startTime) / 1000);
        const hours = Math.floor(diff / 3600);
        const minutes = Math.floor((diff % 3600) / 60);
        const seconds = diff % 60;

        document.getElementById('training-time').textContent =
            `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }, 1000);
}

// Initialize chart on page load
document.addEventListener('DOMContentLoaded', function() {
    initChart();
});

// ==================== Add training control functionality ====================
document.getElementById('start-btn').addEventListener('click', function() {
    fetch('/control/start')
        .then(response => response.json())
        .then(data => {
            const feedback = document.getElementById('control-feedback');
            feedback.className = 'mt-3 text-center alert alert-' +
                                (data.status === 'success' ? 'success' : data.status === 'error' ? 'danger' : 'info');
            feedback.textContent = data.message;
        });
});

document.getElementById('pause-btn').addEventListener('click', function() {
    fetch('/control/pause')
        .then(response => response.json())
        .then(data => {
            const feedback = document.getElementById('control-feedback');
            feedback.className = 'mt-3 text-center alert alert-' +
                                (data.status === 'success' ? 'warning' : 'danger');
            feedback.textContent = data.message;
        });
});

document.getElementById('stop-btn').addEventListener('click', function() {
    fetch('/control/stop')
        .then(response => response.json())
        .then(data => {
            const feedback = document.getElementById('control-feedback');
            feedback.className = 'mt-3 text-center alert alert-danger';
            feedback.textContent = data.message;
        });
});