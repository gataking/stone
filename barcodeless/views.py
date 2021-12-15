from django.shortcuts import render, redirect
from django.http import HttpResponse
from barcodeless.models import Image
# Create your views here.
from barcodeless.prediction import pred_yolo

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
    


def upload(request):
    if request.method == "POST":
        try:
            form=Image()
            form.images = request.FILES['chooseFile']
            form.save()
            print("save")
        except:
            print("save Fail")

        image_path = f"C:/Users/user/Desktop/workspace/stone/media/{str(form.images)}"
        print("!"*30)
        print(image_path)
        item = pred_yolo.predict(image_path)
        print(item)
        context = {
            "items" : item,
        }

        return render(request, 'barcodeless/main.html', context)


def pay(request):
    return render(
        request,
        'barcodeless/pay.html'
    )

def main(request):
    context = {
            "items" : [["누가바", 400, 3], ["빵또아", 1000, 2]],
            # "items" : [["누가바", 400, 4]],
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

def payment(request):
    return render(
        request,
        'barcodeless/payment.html'
    )

def payload(request):
    return render(
        request,
        'barcodeless/payload.html'
    )

