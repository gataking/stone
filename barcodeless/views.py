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

    context = {
            "items" : [["누가바", 400, 2], ["빵또아", 1000, 3]],
    }
    return render(
        request,
        'barcodeless/main.html',
        context
    )

def font(request):
    return render(
        request,
        'barcodeless/font.html'
    )

def pay(request):
    return render(
        request,
        'barcodeless/pay.html'
    )