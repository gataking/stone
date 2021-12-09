from django.urls import path
from . import views

urlpatterns = [
    path('', views.start),
    path('info/', views.info),
    path('start/', views.start),
    path('cam/', views.cam),
    path('pay/', views.pay),
    path('main/', views.main),
    path('font/', views.font), 
    path('pay/', views.pay),
]