import pandas as pd
import numpy as np
import os
import subprocess
from pathlib import Path
import librosa
import cv2
from PIL import Image
from torchvision import transforms
import torch
#pip install opencv-python
#pip install torch
#pip install torchvision


def load_and_validate_excel(excel_path, video_dir):

    df = pd.read_excel(excel_path)

    print(f"\nСтолбцы: {df.columns.tolist()}")

    print("\nПропуски")
    print(df.isnull().sum())

    video_dir = Path(video_dir)
    df['video_exists'] = df['video_file'].apply(lambda x: (video_dir / x).exists())

    found = df['video_exists'].sum()
    print(f"\nВсего видеофайлов: {found} из {len(df)}")

    return df, video_dir

def extract_audio_from_video(video_path, output_dir):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    audio_path = output_dir / f"{video_path.stem}.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        str(audio_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return str(audio_path)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка извлечения аудио из {video_path.name}: {e}")
        return None


def extract_all_audio(df, video_dir, audio_dir):

    audio_dir = Path(audio_dir)
    audio_dir.mkdir(exist_ok=True)

    df['audio_file'] = df['video_file'].apply(
        lambda x: extract_audio_from_video(video_dir / x, audio_dir)
    )

    extracted = df['audio_file'].notna().sum()
    print(f"\nУспешно извлечено аудио: {extracted} из {len(df)}")

    return df


def extract_audio_features(audio_path):
    if audio_path is None:
        return None

    try:
        y, sr = librosa.load(audio_path, sr=16000)

        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500)
        f0_mean = np.nanmean(f0[f0 > 0]) if np.any(f0 > 0) else 0
        f0_std = np.nanstd(f0[f0 > 0]) if np.any(f0 > 0) else 0

        duration = len(y) / sr

        rms = librosa.feature.rms(y=y)
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        pauses = librosa.effects.split(y, top_db=20)
        num_pauses = len(pauses)
        pause_ratio = sum([end - start for start, end in pauses]) / len(y) if len(y) > 0 else 0

        return {
            'f0_mean': f0_mean,
            'f0_std': f0_std,
            'duration': duration,
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'mfcc_mean': mfcc_mean.tolist(),
            'mfcc_std': mfcc_std.tolist(),
            'num_pauses': num_pauses,
            'pause_ratio': pause_ratio
        }

    except Exception as e:
        print(f"Ошибка обработки аудио {audio_path}: {e}")
        return None


def extract_all_audio_features(df):

    df['audio_features'] = df['audio_file'].apply(extract_audio_features)

    extracted = df['audio_features'].notna().sum()
    print(f"\nУспешно анализировано: {extracted} из {len(df)}")

    return df


def extract_video_features(video_path, num_frames=10):
    if video_path is None:
        return None

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        features_list = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)

                preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])

                input_tensor = preprocess(frame_pil).unsqueeze(0)
                features_list.append(input_tensor)

        cap.release()

        if len(features_list) > 0:
            return torch.cat(features_list, dim=0)
        else:
            return None

    except Exception as e:
        print(f"Ошибка обработки видео {video_path}: {e}")
        return None


def extract_all_video_features(df, video_dir):

    df['video_features'] = df['video_file'].apply(
        lambda x: extract_video_features(video_dir / x)
    )
    extracted = df['video_features'].notna().sum()
    print(f"\nУспешно анализировано: {extracted} из {len(df)}")
    return df



def main():
    EXCEL_PATH = "first_annotated_data/annotation.xlsx"
    VIDEO_DIR = "processed_russian_speech_dataset/video_with_audio"
    AUDIO_DIR = "processed_russian_speech_dataset/paired_wav"
    OUTPUT_PATH = "prepared_data.pkl"

    df, video_dir = load_and_validate_excel(EXCEL_PATH, VIDEO_DIR)

    df = extract_all_audio(df, video_dir, AUDIO_DIR)

    df = extract_all_audio_features(df)

    df = extract_all_video_features(df, video_dir)

    import pickle
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(df, f)

if __name__ == "__main__":
    main()