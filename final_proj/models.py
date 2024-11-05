from django.db import models

# Create your models here.

class StockData(models.Model):
    symbol = models.CharField(max_length=10)  # Stock symbol (e.g., AAPL)
    date = models.DateField()                 # Date of the data point
    open_price = models.FloatField()          # Open price
    high_price = models.FloatField()          # High price
    low_price = models.FloatField()           # Low price
    close_price = models.FloatField()         # Close price
    volume = models.BigIntegerField()         # Trading volume
    dividends = models.FloatField(default=0)  # Dividends, if available
    stock_splits = models.FloatField(default=0)  # Stock splits, if available
    upside_downside = models.FloatField(null=True)  # Percentage


    def __str__(self):
        return f"{self.symbol} - {self.date}"