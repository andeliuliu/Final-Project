# final_proj/views.py

# final_proj/views.py

import json
from django.shortcuts import render
from .forms import StockForm
from .services import predict_stock_price

def predict_view(request):
    if request.method == 'POST':
        form = StockForm(request.POST)
        if form.is_valid():
            symbol = form.cleaned_data['symbol'].upper()
            try:
                context = predict_stock_price(symbol)
                context['symbol'] = symbol

                # Serialize data to JSON
                context['dates_json'] = json.dumps(context['dates'])
                context['actual_json'] = json.dumps(context['actual'])
                context['predicted_json'] = json.dumps(context['predicted'])

                return render(request, 'final_proj/results.html', context)
            except Exception as e:
                error_message = 'An error occurred while processing your request. Please ensure the stock symbol is valid and try again.'
                form.add_error(None, error_message)
        else:
            # If form is not valid, render the form with errors
            return render(request, 'final_proj/predict.html', {'form': form})
    else:
        form = StockForm()
    return render(request, 'final_proj/predict.html', {'form': form})