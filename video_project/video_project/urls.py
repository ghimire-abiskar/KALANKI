from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from django.shortcuts import render  # ✅ Add this if missing
from django.contrib import admin  # ✅ Add this if using admin.site

from video_upload.views import upload_video
# Define a simple home view
def home(request):
    return render(request, "home.html")  # Render home.html template

urlpatterns = [
    path("", home, name="home"),  # ✅ Home page route
    path("admin/", admin.site.urls),
    path("upload/", include("video_upload.urls")),  # ✅ Updated route for video upload
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)