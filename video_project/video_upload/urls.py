from django.urls import path
from .views import upload_page

urlpatterns = [
    path("", upload_page, name="upload_page"),
]
