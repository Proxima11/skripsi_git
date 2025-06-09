import torch
import cv2
import numpy as np
import torchvision.transforms as T
import subprocess
import os
import librosa
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
from pydub.effects import normalize

def convert_and_enhance_audio(mp4_path, output_wav_path):
    temp_raw_wav = output_wav_path.replace('.wav', '_raw.wav')
    temp_clean_wav = output_wav_path.replace('.wav', '_tmp.wav')

    try:
        subprocess.run(['ffmpeg', '-y', '-i', mp4_path, temp_raw_wav], check=True)
    except Exception as e:
        print(f"{mp4_path}: audio tidak terdeteksi.")
        return None

    y, sr = librosa.load(temp_raw_wav, sr=None)
    noise_sample = y[0:int(sr * 1)]
    enhanced = nr.reduce_noise(y=y, y_noise=noise_sample, sr=sr, prop_decrease=1.0)
    sf.write(temp_clean_wav, enhanced, sr)

    audio = AudioSegment.from_wav(temp_clean_wav)
    normalized = normalize(audio)
    normalized.export(output_wav_path, format='wav')

    os.remove(temp_raw_wav)
    os.remove(temp_clean_wav)
    print(f"✅ Audio diproses: {output_wav_path}")
    return output_wav_path

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor()
])

def extract_mouth(frame, landmarks):
    x = landmarks[48:68, 0]
    y = landmarks[48:68, 1]
    x1, x2 = np.min(x), np.max(x)
    y1, y2 = np.min(y), np.max(y)
    pad = 10
    return frame[y1-pad:y2+pad, x1-pad:x2+pad]

def extract_mouth_frames(video_path, fps=5):
    import dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("mlapp/ml_model/shape_predictor_68_face_landmarks.dat")

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    target_frame_idxs = np.linspace(0, total_frames - 1, int(duration * fps), dtype=int)

    frames = []
    idx = 0
    current_frame = 0
    last_valid_mouth = None

    while cap.isOpened() and idx < len(target_frame_idxs):
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame == target_frame_idxs[idx]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if len(faces) > 0:
                shape = predictor(gray, faces[0])
                landmarks = np.array([[p.x, p.y] for p in shape.parts()])
                mouth = extract_mouth(frame, landmarks)

                if mouth is not None and mouth.size > 0:
                    mouth_tensor = transform(mouth)
                    last_valid_mouth = mouth_tensor.clone()
                    frames.append(mouth_tensor)
                elif last_valid_mouth is not None:
                    frames.append(last_valid_mouth.clone())
        
            elif last_valid_mouth is not None:
                frames.append(last_valid_mouth.clone())

            idx += 1

        current_frame += 1

    cap.release()
    return frames


def extract_mel_spectrogram(audio_path, target_frames, sr_target=16000, n_mels=128):
    y, sr = librosa.load(audio_path, sr=sr_target)
    duration = target_frames / 5  # 5 fps
    max_len = int(duration * sr_target)

    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)), 'constant')

    S = librosa.stft(y, n_fft=2048, hop_length=512)
    S_power = np.abs(S) ** 2
    mel = librosa.feature.melspectrogram(S=S_power, sr=sr_target, n_mels=n_mels)

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_tensor = torch.tensor(mel_db).unsqueeze(0)
    return mel_tensor

def interpolate_mel(mel_tensor, target_T):
    import torch.nn.functional as F
    mel_tensor = mel_tensor.unsqueeze(0)
    mel_resized = F.interpolate(mel_tensor, size=(mel_tensor.shape[2], target_T), mode='bilinear', align_corners=False)
    return mel_resized.squeeze(0) 


def preprocess_video_to_pt(video_path, fps=5):
    audio_enhanced_path = video_path.replace('.mp4', '_enhanced.wav')

    final_audio_path = convert_and_enhance_audio(video_path, audio_enhanced_path)
    if final_audio_path is None:
        return {'error': 'Audio tidak terdeteksi'}
    
    frames = extract_mouth_frames(video_path)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = frame_count / video_fps
    cap.release()
    
    expected_min_frame_count = int(duration_sec * fps)
    
    if len(frames) < expected_min_frame_count:
        return {'error': 'Bibir tidak terdeteksi'}
    
    frames_tensor = torch.stack(frames)
    T_bibir = frames_tensor.shape[0]
    
    mel_tensor = extract_mel_spectrogram(final_audio_path, target_frames=T_bibir)
    mel_tensor = interpolate_mel(mel_tensor, target_T=T_bibir)
    
    T_audio = mel_tensor.shape[-1]
    if T_bibir != T_audio:
        return {'error': 'Audio tidak terdeteksi'}
    
    if mel_tensor.max() > 1.0 or mel_tensor.min() < 0.0:
        mel_tensor = torch.clamp(mel_tensor, min=-80, max=0) 
        mel_tensor = (mel_tensor + 80) / 80
    
    return {
        'frames': frames_tensor,
        'mel': mel_tensor
    }
    
def lipsync_detector_single(model, frames_tensor, mel_tensor, device, class_names=None):
    import torch
    import torch.nn.functional as F

    model.eval()
    frames_tensor = frames_tensor.to(device)
    mel_tensor = mel_tensor.to(device)

    with torch.no_grad():
        output = model(frames_tensor, mel_tensor)
        prob_tensor = F.softmax(output, dim=1)
        pred_idx = torch.argmax(prob_tensor, dim=1).item()
        prob = prob_tensor[0, pred_idx].item()
        prob_all = prob_tensor.cpu().numpy()[0].tolist()

    pred_label = class_names[pred_idx] if class_names else pred_idx
    return pred_label, prob, prob_all

def lip_sync_detector(video_path):
    import torch
    import torch.nn.functional as F
    from .model_loader import load_lipsync_models
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_lipsync_models().to(device)
    
    CLASS_NAMES = ["REAL", "LIP-SYNC DEEPFAKE"]
    
    data = preprocess_video_to_pt(video_path)
    
    if data is None:
        return {'success': False, 'error': 'Preprocessing gagal tanpa informasi'}
    if 'error' in data:
        return {'success': False, 'error': f'Preprocessing gagal: {data["error"]}'}
    
    frames_tensor = data['frames'].unsqueeze(0)
    mel_tensor = data['mel'].unsqueeze(0)
    
    label, conf, _ = lipsync_detector_single(model, frames_tensor, mel_tensor, device, CLASS_NAMES)
    
    try:
        base_path = video_path.replace('.mp4', '')
        for suffix in ['_raw.wav', '_tmp.wav', '_enhanced.wav']:
            audio_file = f"{base_path}{suffix}"
            if os.path.exists(audio_file):
                os.remove(audio_file)
                print(f"[CLEANUP] Hapus file: {audio_file}")
    except Exception as e:
        print(f"[WARNING] Gagal menghapus audio: {e}")
        
    return {
        'success': True,
        'label': CLASS_NAMES.index(label),
        'label_name': label,
        'confidence': conf
    }