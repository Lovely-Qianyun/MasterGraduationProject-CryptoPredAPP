document.addEventListener('DOMContentLoaded', function() {
    let currentCrypto = null;
    let currentIndicator = null;
    let currentPeriod = '1y';

    document.getElementById('data-update-time').textContent = new Date().toLocaleDateString();
    updateMatrix(currentPeriod);

    document.querySelectorAll('[data-period]').forEach(button => {
        button.addEventListener('click', function() {
            document.querySelectorAll('[data-period]').forEach(btn => {
                btn.classList.remove('active');
            });
            this.classList.add('active');
            const period = this.getAttribute('data-period');
            currentPeriod = period;
            updateMatrix(period);
        });
    });

    function updateMatrix(period) {
        document.getElementById('correlation-matrix').innerHTML = '<p>Loading...</p>';
        fetch(`/api/correlations?period=${period}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response error');
                }
                return response.json();
            })
            .then(data => {
                renderCorrelationMatrix(data);
                if (currentCrypto && currentIndicator) {
                    loadHistoricalCorrelations(currentCrypto, currentIndicator, period);
                }
            })
            .catch(error => {
                console.error('Data loading failed:', error);
                document.getElementById('correlation-matrix').innerHTML = '<p>Data loading failed, please try again</p>';
            });
    }

    function renderCorrelationMatrix(correlations) {
        const matrix = document.getElementById('correlation-matrix');
        matrix.innerHTML = '';

        const macroIndicators = [
            'fed_rate', 'treasury_yield', 'dxy', 'cpi', 'vix', 'sp500'
        ];

        const indicatorNames = {
            'fed_rate': 'Fed Rate',
            'treasury_yield': 'Treasury Yield',
            'dxy': 'Dollar Index',
            'cpi': 'Inflation CPI',
            'vix': 'VIX Fear Index',
            'sp500': 'S&P 500 Index'
        };

        const headerRow = document.createElement('div');
        headerRow.className = 'matrix-header';

        const emptyCell = document.createElement('div');
        emptyCell.textContent = '';
        headerRow.appendChild(emptyCell);

        macroIndicators.forEach(indicator => {
            const cell = document.createElement('div');
            cell.textContent = indicatorNames[indicator];
            cell.title = getIndicatorDescription(indicator);
            headerRow.appendChild(cell);
        });

        matrix.appendChild(headerRow);

        Object.keys(correlations).forEach(crypto => {
            const row = document.createElement('div');
            row.className = 'matrix-row';

            const cryptoCell = document.createElement('div');
            cryptoCell.textContent = crypto;
            row.appendChild(cryptoCell);

            macroIndicators.forEach(indicator => {
                const cell = document.createElement('div');
                const value = correlations[crypto][indicator];
                const absValue = Math.abs(value);
                let strengthClass = '';

                if (absValue > 0.6) {
                    strengthClass = value > 0 ? 'positive-4' : 'negative-4';
                } else if (absValue > 0.4) {
                    strengthClass = value > 0 ? 'positive-3' : 'negative-3';
                } else if (absValue > 0.2) {
                    strengthClass = value > 0 ? 'positive-2' : 'negative-2';
                } else {
                    strengthClass = value > 0 ? 'positive-1' : 'negative-1';
                }

                cell.className = strengthClass;
                cell.innerHTML = `
                    <span class="correlation-dot ${strengthClass}"></span>
                    ${value.toFixed(2)}
                `;

                cell.addEventListener('click', () => {
                    showIndicatorImpact(crypto, indicator, value, currentPeriod);
                });

                row.appendChild(cell);
            });

            matrix.appendChild(row);
        });
    }

    function getIndicatorDescription(indicator) {
        const descriptions = {
            'fed_rate': 'Federal Reserve Rate: Influences global capital flows',
            'treasury_yield': 'Treasury Yield: Represents risk-free interest rate',
            'dxy': 'Dollar Index: Measures USD exchange rate changes',
            'cpi': 'Inflation CPI: Measures price level changes',
            'vix': 'VIX Fear Index: Reflects investor sentiment',
            'sp500': 'S&P 500 Index: Represents US stock market performance'
        };
        return descriptions[indicator] || '';
    }

    function showIndicatorImpact(crypto, indicator, correlation, period) {
        currentCrypto = crypto;
        currentIndicator = indicator;

        const indicatorNameMap = {
            'fed_rate': 'Fed Rate',
            'treasury_yield': 'Treasury Yield',
            'dxy': 'Dollar Index',
            'cpi': 'Inflation CPI',
            'vix': 'VIX Fear Index',
            'sp500': 'S&P 500 Index'
        };

        document.getElementById('indicator-title').textContent = `${crypto} & ${indicatorNameMap[indicator]} Correlation`;
        document.getElementById('indicator-description').textContent = getIndicatorDescription(indicator);
        document.getElementById('indicator-name').textContent = indicatorNameMap[indicator];
        document.getElementById('crypto-name').textContent = crypto;

        const priceChange = (correlation * 0.8).toFixed(2);
        document.getElementById('price-change').textContent = `${priceChange}%`;

        document.getElementById('impact-analysis').classList.remove('d-none');
        loadHistoricalCorrelations(crypto, indicator, period);
    }

    function loadHistoricalCorrelations(crypto, indicator, period) {
        const chartContainer = document.getElementById('correlation-chart');
        chartContainer.innerHTML = '<p>Loading historical data...</p>';

        fetch(`/api/historical-correlations?crypto=${crypto}&indicator=${indicator}&period=${period}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response error');
                }
                return response.json();
            })
            .then(data => {
                renderCorrelationChart(crypto, indicator, data);
            })
            .catch(error => {
                console.error('Historical data loading failed:', error);
                document.getElementById('correlation-matrix').innerHTML =
                    `<p>Data loading failed, please try again</p>
<p>Error details: ${error.message}</p>`;
            });
    }

    function renderCorrelationChart(crypto, indicator, historicalData) {
        const ctx = document.getElementById('correlation-chart').getContext('2d');
        const canvas = document.createElement('canvas');
        canvas.id = 'correlation-chart-canvas';
        document.getElementById('correlation-chart').innerHTML = '';
        document.getElementById('correlation-chart').appendChild(canvas);
        const ctxNew = canvas.getContext('2d');

        if (window.correlationChart) {
            window.correlationChart.destroy();
        }

        const labels = historicalData.map(item => {
            const date = new Date(item.date);
            return `${date.getFullYear()}-${(date.getMonth() + 1).toString().padStart(2, '0')}`;
        });

        const dataPoints = historicalData.map(item => item.correlation);

        window.correlationChart = new Chart(ctxNew, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: `${crypto} & ${indicator} Correlation`,
                    data: dataPoints,
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    borderWidth: 2,
                    pointRadius: 3,
                    pointBackgroundColor: 'rgb(54, 162, 235)',
                    tension: 0.1,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `Historical Correlation: ${crypto} & ${indicator}`,
                        font: {
                            size: 16
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Correlation: ${context.parsed.y.toFixed(4)}`;
                            },
                            title: function(context) {
                                return context[0].label;
                            }
                        }
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        min: -1,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Correlation',
                            font: {
                                weight: 'bold'
                            }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Date',
                            font: {
                                weight: 'bold'
                            }
                        },
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                }
            }
        });
    }
});