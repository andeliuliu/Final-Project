from django.shortcuts import render
from .forms import StockForm
from .services import predict_stock_price
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView, View
from django.http import JsonResponse
from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from .models import StockData
from .services import predict_stock_price
from datetime import date


from .models import *
from .forms import *
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

class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'final_proj/dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        watchlist = StockData.objects.filter(user=self.request.user)

        # Fetch prediction data dynamically
        enriched_watchlist = []
        for stock in watchlist:
            prediction_data = predict_stock_price(stock.symbol, 'Close', 1)
            enriched_watchlist.append({
                'symbol': stock.symbol,
                'current_price': prediction_data.get('current_price', 'N/A'),
                'upside_downside': prediction_data.get('upside_downside', 'N/A'),
                'recommendation': prediction_data.get('recommendation', 'N/A'),
            })

        context.update({
            'user': self.request.user,
            'watchlist': enriched_watchlist,
        })
        return context

@login_required
def add_to_watchlist(request):
    if request.method == 'POST':
        symbol = request.POST.get('symbol')
        target = 'Close'  
        horizon = int(request.POST.get('horizon', 1))

        try:
            # Fetch prediction data
            prediction_data = predict_stock_price(symbol, target, horizon)

            # Extract required fields
            current_price = prediction_data.get('current_price')
            upside_downside = prediction_data.get('upside_downside')
            recommendation = prediction_data.get('recommendation')  

            # Check for duplicates
            existing_stock = StockData.objects.filter(
                user=request.user, symbol=symbol, date=date.today()
            ).first()

            if existing_stock:
                messages.info(request, f"{symbol} is already in your watchlist.")
            else:
                # Create a new stock record with the current date
                StockData.objects.create(
                    user=request.user,
                    symbol=symbol,
                    close_price=current_price,
                    upside_downside=upside_downside,
                    date=date.today(),

                )
                messages.success(
                    request, 
                    f"{symbol} was successfully added to your watchlist with recommendation: {recommendation}."
                )

        except Exception as e:
            print(f"Error during prediction: {e}")
            messages.error(request, "An error occurred while fetching stock data.")

        return redirect('dashboard')

    messages.error(request, "Invalid request method.")
    return redirect('dashboard')

# Register view for new users
class RegisterView(View):
    def get(self, request):
        form = RegisterForm()
        return render(request, 'final_proj/register.html', {'form': form})

    def post(self, request):
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            login(request, user)
            return redirect('dashboard')
        return render(request, 'final_proj/register.html', {'form': form})
