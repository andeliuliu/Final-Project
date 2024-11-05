

from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_view, name='home'),  # Home page for the form
    path('predict/', views.predict_view, name='predict'),  # Separate URL for predictions
]