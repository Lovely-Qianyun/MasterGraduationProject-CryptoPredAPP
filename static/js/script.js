// DOM elements
const form = document.getElementById('predictionForm');
const predictBtn = document.getElementById('predictBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultContainer = document.getElementById('resultContainer');

// Form submission handler
form.addEventListener('submit', async function (e) {
    e.preventDefault();

    // Get form data
    const formData = new FormData(form);
    const cryptocurrency = formData.get('cryptocurrency');
    const model = formData.get('model');
    const prediction_days = formData.get('prediction_days');

    // Validate form
    if (!cryptocurrency || !model || !prediction_days) {
        showError('Please fill in all required fields');
        return;
    }

    // Show loading state
    showLoading();

    try {
        // Send prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                cryptocurrency: cryptocurrency,
                model: model,
                prediction_days: parseInt(prediction_days)
            })
        });

        const result = await response.json();

        if (response.ok) {
            showResult(result);
        } else {
            showError(result.error || 'Prediction failed, please try again');
        }
    } catch (error) {
        console.error('Prediction request failed:', error);
        showError('Network error, please check connection and try again');
    }
});

// Show loading state
function showLoading() {
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Predicting...';
    loadingIndicator.style.display = 'block';
    resultContainer.innerHTML = '';
}

// Show prediction results
function showResult(result) {
    hideLoading();

    const resultHTML = `
        <div class="prediction-result">
            <img src="${result.plot_url}" alt="Prediction Chart" class="prediction-chart" />
            <div class="prediction-info">
                <h3><i class="fas fa-info-circle"></i> Prediction Information</h3>
                <p><strong>Cryptocurrency:</strong> ${result.cryptocurrency}</p>
                <p><strong>Model Used:</strong> ${result.model}</p>
                <p><strong>Prediction Period:</strong> ${result.prediction_days} days</p>
                <p><strong>Prediction Time:</strong> ${new Date().toLocaleString('en-US')}</p>
                
                <h4 style="margin-top: 15px;"><i class="fas fa-chart-line"></i> Latest Predicted Prices</h4>
                <div class="prediction-summary">
                    ${result.predictions.prices.slice(0, 5).map((price, index) => `
                        <p><strong>Day ${index + 1} (${result.predictions.dates[index]}):</strong> $${price.toFixed(2)}</p>
                    `).join('')}
                    ${result.predictions.prices.length > 5 ? '<p>...</p>' : ''}
                </div>
            </div>
        </div>
    `;

    resultContainer.innerHTML = resultHTML;
}

// Show error message
function showError(message) {
    hideLoading();

    const errorHTML = `
        <div class="error-message">
            <i class="fas fa-exclamation-triangle"></i>
            <strong>Error:</strong> ${message}
        </div>
    `;

    resultContainer.innerHTML = errorHTML;
}

// Hide loading state
function hideLoading() {
    predictBtn.disabled = false;
    predictBtn.innerHTML = '<i class="fas fa-magic"></i> Start Prediction';
    loadingIndicator.style.display = 'none';
}

// Form validation
document.querySelectorAll('select').forEach(select => {
    select.addEventListener('change', function () {
        this.style.borderColor = this.value ? '#27ae60' : '#e1e8ed';
    });
});

// Page initialization after loading
document.addEventListener('DOMContentLoaded', function () {
    // Add page loading animation
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.5s ease-in-out';

    setTimeout(() => {
        document.body.style.opacity = '1';
    }, 100);

    // Add hover effects to select boxes
    document.querySelectorAll('select').forEach(select => {
        select.addEventListener('mouseenter', function () {
            if (!this.value) {
                this.style.borderColor = '#3498db';
            }
        });

        select.addEventListener('mouseleave', function () {
            if (!this.value) {
                this.style.borderColor = '#e1e8ed';
            }
        });
    });
});

// Keyboard shortcut support
document.addEventListener('keydown', function (e) {
    // Ctrl+Enter shortcut to submit form
    if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        if (!predictBtn.disabled) {
            form.dispatchEvent(new Event('submit'));
        }
    }
});

// Add responsive handling
function handleResize() {
    // Adjust layout on mobile devices
    if (window.innerWidth <= 768) {
        document.querySelectorAll('.card').forEach(card => {
            card.style.margin = '0 0 20px 0';
        });
    }
}

window.addEventListener('resize', handleResize);
handleResize(); // Initial call
