# final_proj/forms.py

from django import forms

class StockForm(forms.Form):
    symbol = forms.CharField(
        max_length=10,
        label='Stock Symbol',
        widget=forms.TextInput(attrs={'placeholder': 'Ex: AAPL, TSLA'})
    )