# urls.py

from django.urls import path
from .views import home, detect_fraud, result,front,index

urlpatterns = [
    path('', home, name='home'),
    path('front/', front,name = 'front'),
    path('detect_fraud/', detect_fraud, name='detect_fraud'),
    path('index/', index, name='index'),
    path('result/<str:is_fraud>/', result, name='result'),
]
