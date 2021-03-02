from django.urls import path
from digits import views

urlpatterns = [
    path('digits/', views.Digits.as_view())
]