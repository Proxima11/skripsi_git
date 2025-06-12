import cv2
import dlib
import torch
import numpy as np
from imutils import face_utils

def preprocess_video_to_pt(video_path, fps_target=5, output_face_size=(224, 224)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    detector = dlib.get_frontal_face_detector()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_frames / fps_video if fps_video > 0 else 0

    if total_frames <= 0 or fps_video <= 0:
        print(f"❌ Tidak bisa membaca video: {video_path}")
        cap.release()
        return {'error': 'Wajah tidak terdeteksi'}

    frame_interval = int(round(fps_video / fps_target))
    current_frame = 0
    last_valid_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 1)

            if len(faces) > 0:
                (x, y, w, h) = face_utils.rect_to_bb(faces[0])
                x, y = max(0, x), max(0, y)
                face_crop = frame[y:y+h, x:x+w]
                last_valid_frame = face_crop
            elif last_valid_frame is not None:
                face_crop = last_valid_frame.copy()
            else:
                current_frame += 1
                continue

            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_crop = cv2.resize(face_crop, output_face_size)
            face_crop = face_crop / 255.0
            frames.append(torch.tensor(face_crop, dtype=torch.float32).permute(2, 0, 1))

        current_frame += 1

    cap.release()

    expected_min_frame_count = int(duration_sec * fps_target)
    if len(frames) < expected_min_frame_count:
        return {'error': 'Wajah tidak terdeteksi'}

    return {
        'frames': torch.stack(frames)
    }


def visual_detector_single(model, frames_tensor, device, class_names=None):
    import torch
    import torch.nn.functional as F

    model.eval()
    frames_tensor = frames_tensor.to(device)

    with torch.no_grad():
        output = model(frames_tensor)
        prob_tensor = F.softmax(output, dim=1)
        pred_idx = torch.argmax(prob_tensor, dim=1).item()
        prob = prob_tensor[0, pred_idx].item()
        prob_all = prob_tensor.cpu().numpy()[0].tolist()

    pred_label = class_names[pred_idx] if class_names else pred_idx
    return pred_label, prob, prob_all

def visual_detector(video_path):
    import torch
    import torch.nn.functional as F
    from .model_loader import load_visual_models
    
    device = torch.device("cpu")
    model = load_visual_models().to(device)
    CLASS_NAMES = ["REAL", "FACE-SWAP DEEPFAKE"]
    
    data = preprocess_video_to_pt(video_path)
    
    if data is None:
        return {'success': False, 'error': 'Preprocessing gagal tanpa informasi'}
    if 'error' in data:
        return {'success': False, 'error': f'Preprocessing gagal: {data["error"]}'}
     
    frames = data['frames'].unsqueeze(0)
    
    label, conf, _ = visual_detector_single(model, frames, device, CLASS_NAMES)
    return {
        'success': True,
        'label': CLASS_NAMES.index(label),
        'label_name': label,
        'confidence': conf
    }