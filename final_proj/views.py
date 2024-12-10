from django.shortcuts import render
from .forms import StockForm
from .services import predict_stock_price
import json

def predict_view(request):
    if 'symbol' in request.GET:
        symbol = request.GET.get('symbol').upper()
        horizon = request.GET.get('horizon')
        if horizon:
            horizon = int(horizon)

        try:
            context = predict_stock_price(symbol, 'Close', horizon)
            context['symbol'] = symbol
            context['horizon'] = horizon

            context['dates_json'] = json.dumps(context['dates'])
            context['actual_json'] = json.dumps(context['actual'])
            context['predicted_json'] = json.dumps(context['predicted'])

            return render(request, 'final_proj/results.html', context)
        except Exception as e:
            error_message = 'An error occurred while processing your request. Please ensure the stock symbol is valid and try again.'
            print(f"Error: {e}")
            return render(request, 'final_proj/predict.html', {'form': StockForm(), 'error_message': error_message})

    elif request.method == 'POST':
        form = StockForm(request.POST)
        if form.is_valid():
            symbol = form.cleaned_data['symbol'].upper()
            return render(request, 'final_proj/results.html', {'symbol': symbol, 'horizon': None})
        else:
            return render(request, 'final_proj/predict.html', {'form': form})

    else:
        form = StockForm()
        return render(request, 'final_proj/predict.html', {'form': form})
