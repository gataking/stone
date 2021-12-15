from django.urls import path
from . import views

urlpatterns = [
    path('', views.start),
    path('info/', views.info),
    path('start/', views.start),
    path('cam/', views.cam),
    path('upload/', views.upload),
    path('pay/', views.pay),
    path('main/', views.main),
    path('font/', views.font), 
    path('pay/', views.pay),
    path('payment/', views.payment),
    path('payload/', views.payload),

]