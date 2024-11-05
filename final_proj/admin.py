from django.contrib import admin
from .models import StockData

# Register your models here.
class StockDataAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume')

# Register the StockData model with the custom admin view
admin.site.register(StockData, StockDataAdmin)