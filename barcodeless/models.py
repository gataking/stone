from django.db import models

# Create your models here.
class Image(models.Model):
    images = models.ImageField(upload_to = "images/") 
    # settings.py에 명시된 MEDIA_URL(/media/) 안에 images라는 폴더를 만들어 그 안에 저장
