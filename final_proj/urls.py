
# final_proj/urls.py
from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.predict_view, name='home'),  # Home page for the form
    path('predict/', views.predict_view, name='predict'),  # Separate URL for predictions
    path('login/', auth_views.LoginView.as_view(template_name='final_proj/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path('register/', views.RegisterView.as_view(), name='register'),
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),
    path('add_to_watchlist/', views.add_to_watchlist, name='add_to_watchlist'),
]