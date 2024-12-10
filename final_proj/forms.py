# final_proj/forms.py

from django import forms
from django.contrib.auth.models import User


class StockForm(forms.Form):
    symbol = forms.CharField(
        max_length=10,
        label='Stock Symbol',
        widget=forms.TextInput(attrs={'placeholder': 'Ex: AAPL, TSLA'})
    )

class RegisterForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['username', 'password', 'first_name', 'last_name', 'email']
        widgets = {
            'password': forms.PasswordInput(),
        }
