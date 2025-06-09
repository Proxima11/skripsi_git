from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse


from .visual_detector import visual_detector
from .lipsync_detector import lip_sync_detector
import os

previous_video_path = None

def upload_file(request):
    global previous_video_path
    
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage()
        
        # Hapus video sebelumnya
        if previous_video_path and os.path.exists(previous_video_path):
            try:
                os.remove(previous_video_path)
                print(f"[INFO] Video sebelumnya dihapus: {previous_video_path}")
            except Exception as e:
                print(f"[WARNING] Gagal menghapus video sebelumnya: {e}")
        
        file_path = fs.save(uploaded_file.name, uploaded_file)
        full_path = fs.path(file_path)
        previous_video_path = full_path

        try:
            lip_result = lip_sync_detector(full_path)
            
            if not lip_result['success']:
                result = lip_result['error']
            elif lip_result['label'] == 1:
                result = "LIP-SYNC DEEPFAKE"
            else:
                visual_result = visual_detector(full_path)
            
                if not visual_result['success']:
                    result = visual_result['error']
                elif visual_result['label'] == 1:
                    result = "FACE-SWAP DEEPFAKE"
                else:
                    result = "REAL"

        except Exception as e:
            result = f"Error during prediction: {str(e)}"
            
        return JsonResponse({'result': result})

    return render(request, 'mlapp/upload.html')


def result_view(request):
    result = request.session.get('result', 'Tidak ada hasil.')
    return render(request, "mlapp/result.html", {'result': result})

