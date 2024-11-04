// results.js

// The data variables will be defined in the HTML template
const ctx = document.getElementById('priceChart').getContext('2d');

const chartData = {
    labels: dates,
    datasets: [
        {
            label: 'Actual Prices',
            data: actualPrices,
            borderColor: 'blue',
            fill: false,
        },
        {
            label: 'Predicted Prices',
            data: predictedPrices,
            borderColor: 'red',
            fill: false,
        }
    ]
};

const priceChart = new Chart(ctx, {
    type: 'line',
    data: chartData,
    options: {
        scales: {
            x: {
                display: true,
                title: {
                    display: true,
                    text: 'Date'
                }
            },
            y: {
                display: true,
                title: {
                    display: true,
                    text: 'Price ($)'
                }
            }
        }
    }
});