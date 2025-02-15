from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt
import os
from .id_tracing import run_tracking  # ✅ Import from id_tracing.py

@csrf_exempt
def upload_video(request):
    if request.method == "POST" and request.FILES.get("video"):
        video_file = request.FILES["video"]

        # Save uploaded file
        video_path = default_storage.save("uploads/" + video_file.name, ContentFile(video_file.read()))
        video_full_path = os.path.join("media", video_path)

        # Define output path
        output_filename = f"processed_{video_file.name}"
        output_path = os.path.join("media", "processed_videos", output_filename)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Call ID tracking function
        run_tracking(video_full_path, output_path)

        # Return processed video URL
        processed_video_url = f"/media/processed_videos/{output_filename}"
        return JsonResponse({"message": "File processed successfully!", "video_url": processed_video_url})

    return render(request, "upload.html")
