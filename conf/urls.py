"""conf URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('barcodeless/', include('barcodeless.urls')),
]
# runserver 명령을 통해 구동하는 개발 서버에서는 media 파일을 자동으로 서빙해 주지 않습니다. 
# 따라서 수동으로 urlpattern을 추가함으로서 서빙해야합니다.
# 이 설정을 추가하면 MEDIA_URL로 들어오는 요청에 대해 MEDIA_ROOT 경로를 탐색하게 됩니다.
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
