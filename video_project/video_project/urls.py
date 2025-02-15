from django.urls import path, include

urlpatterns = [
    path("upload/", include("video_upload.urls")),
]
