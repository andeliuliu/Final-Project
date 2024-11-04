from django.db import models

# Create your models here.
class StockData(models.Model):
    ticker = models.CharField(max_length=10)
    date = models.DateField()
    open = models.FloatField()
    close = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    volume = models.BigIntegerField()

    class Meta:
        unique_together = ("ticker", "date")
        ordering = ["-date"]

    def __str__(self):
        return f"{self.ticker} on {self.date}: Close - {self.close}"