from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt  # Disable CSRF for testing; remove in production
def upload_video(request):
    if request.method == "POST":
        # Handle file upload here
        return JsonResponse({"message": "File uploaded successfully!"})
    
    # If GET request, render the upload form
    return render(request, "upload.html")  # Ensure 'upload.html' exists
