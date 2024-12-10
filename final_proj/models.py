from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class StockData(models.Model):
    symbol = models.CharField(max_length=10)  # Stock symbol (e.g., AAPL)
    date = models.DateField(null=True, blank=True)                 # Date of the data point
    open_price = models.FloatField(null=True, blank=True)          # Open price
    high_price = models.FloatField(null=True, blank=True)          # High price
    low_price = models.FloatField(null=True, blank=True)           # Low price
    close_price = models.FloatField(null=True, blank=True)         # Close price
    volume = models.BigIntegerField(null=True, blank=True)         # Trading volume
    dividends = models.FloatField(default=0, blank=True)  # Dividends, if available
    stock_splits = models.FloatField(default=0, blank=True)  # Stock splits, if available
    upside_downside = models.FloatField(null=True, blank=True)  # Percentage

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='stocks')

    def __str__(self):
        return f"{self.symbol} - {self.date} ({self.recommendation})"
    
