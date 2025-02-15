from django.contrib import admin
from django.urls import path, include
from django.shortcuts import render

# Define a simple home view
def home(request):
    return render(request, "home.html")  # Render home.html template

urlpatterns = [
    path("", home, name="home"),  # ✅ Home page route
    path("admin/", admin.site.urls),
    path("upload/", include("video_upload.urls")),  # ✅ Updated route for video upload
]
