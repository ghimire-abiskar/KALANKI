from django.urls import path
from .views import upload_video  # Ensure this matches your function name in views.py

urlpatterns = [
    path("", upload_video, name="upload"),  # ✅ Set as 'upload' for the home page link
]
