from django.shortcuts import render, redirect
from django.http import HttpResponse
# Create your views here.

def info(request):
    return render(
        request,
        'barcodeless/info.html'
    )


def start(request):
    return render(
        request,
        'barcodeless/start.html'
    )


def cam(request):
    return render(
        request,
        'barcodeless/cam.html'
    )


def pay(request):
    return render(
        request,
        'barcodeless/pay.html'
    )

def main(request):
    return render(
        request,
        'barcodeless/main.html'
    )