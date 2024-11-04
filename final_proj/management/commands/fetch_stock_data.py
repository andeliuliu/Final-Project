import yfinance as yf
from django.core.management.base import BaseCommand
from final_proj.models import StockData  # Adjust this import based on your app's structure

class Command(BaseCommand):
    help = "Fetches stock data from Yahoo Finance and saves it to the database"

    def add_arguments(self, parser):
        parser.add_argument("ticker", type=str, help="Stock ticker symbol (e.g., AAPL for Apple)")

    def handle(self, *args, **options):
        ticker_symbol = options["ticker"]
        ticker = yf.Ticker(ticker_symbol)

        # Fetch historical stock data for the past year
        data = ticker.history(period="1y")
        
        # Loop through data and save to database
        for date, row in data.iterrows():
            StockData.objects.update_or_create(
                date=date,
                ticker=ticker_symbol,
                defaults={
                    "open": row["Open"],
                    "close": row["Close"],
                    "high": row["High"],
                    "low": row["Low"],
                    "volume": row["Volume"],
                },
            )

        self.stdout.write(self.style.SUCCESS(f"Stock data for {ticker_symbol} saved successfully"))