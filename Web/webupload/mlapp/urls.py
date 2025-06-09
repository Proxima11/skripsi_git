from django.urls import path
from .views import upload_file, result_view

app_name = "mlapp"  # Tambahkan ini agar namespace 'mlapp' terdaftar

urlpatterns = [
    path('', upload_file, name='upload_file'),
    path("upload/", upload_file, name="upload"),
    path("result/", result_view, name="result"),
]
