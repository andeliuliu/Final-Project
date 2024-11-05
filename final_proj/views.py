# final_proj/views.py

from django.shortcuts import render
from .forms import StockForm
from .services import predict_stock_price
import json

def predict_view(request):
    if request.method == 'POST':
        form = StockForm(request.POST)
        if form.is_valid():
            symbol = form.cleaned_data['symbol'].upper()
            horizon = 1  # Default horizon for now

            try:
                # Call predict_stock_price to get stock data
                context = predict_stock_price(symbol, horizon)
                context['symbol'] = symbol

                # Serialize data for JavaScript
                context['dates_json'] = json.dumps(context.get('dates', []))
                context['actual_json'] = json.dumps(context.get('actual', []))

                # Debug statements to check data
                print("Context Data:", context)

                # Render results with the context data
                return render(request, 'final_proj/results.html', context)
            except Exception as e:
                error_message = f'An error occurred while processing your request: {e}'
                form.add_error(None, error_message)
                print(f"Error: {e}")
                return render(request, 'final_proj/predict.html', {'form': form})

    # Render the form for GET requests or invalid POST
    form = StockForm()
    return render(request, 'final_proj/predict.html', {'form': form})