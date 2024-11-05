from django.shortcuts import render
from .forms import StockForm
from .services import predict_stock_price
import json

def predict_view(request):
    # Check if a symbol and horizon are provided in the GET request
    if 'symbol' in request.GET:
        symbol = request.GET.get('symbol').upper()
        horizon = int(request.GET.get('horizon', 1))  # Default to 1 day if horizon not specified

        try:
            # Fetch prediction data with specified horizon
            context = predict_stock_price(symbol, horizon)
            context['symbol'] = symbol
            context['horizon'] = horizon  # Pass horizon to template

            # Serialize only the actual data to JSON for JavaScript
            context['dates_json'] = json.dumps(context['dates'])
            context['actual_json'] = json.dumps(context['actual'])

            return render(request, 'final_proj/results.html', context)
        except Exception as e:
            error_message = 'An error occurred while processing your request. Please ensure the stock symbol is valid and try again.'
            print(f"Error: {e}")  # Optional debug log
            return render(request, 'final_proj/predict.html', {'form': StockForm(), 'error_message': error_message})

    # Handle the initial POST request with the form submission
    elif request.method == 'POST':
        form = StockForm(request.POST)
        if form.is_valid():
            symbol = form.cleaned_data['symbol'].upper()
            # Redirect to the results page with symbol in the URL query
            return render(request, 'final_proj/results.html', {'symbol': symbol, 'horizon': 1})
        else:
            return render(request, 'final_proj/predict.html', {'form': form})

    # If no GET or POST data, show the prediction form
    else:
        form = StockForm()
        return render(request, 'final_proj/predict.html', {'form': form})