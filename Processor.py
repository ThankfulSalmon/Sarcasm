import os
import sys
import numpy as np
import pandas as pd
import torch
import librosa
import cv2
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer, BertModel

class TextPreprocessor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
        self.model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')
        self.model.eval()

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding='max_length',
            return_attention_mask=True
        )
        return inputs

    def get_embeddings(self, text):
        inputs = self.tokenize_text(text)

        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        return cls_embedding.squeeze().numpy()

    def process_batch(self, texts):
        embeddings = []
        for text in texts:
            emb = self.get_embeddings(text)
            embeddings.append(emb)
        return np.array(embeddings)

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, audio_dir="audio_files"):
        self.sample_rate = sample_rate
        self.audio_dir = Path(audio_dir)

    def get_audio_path(self, video_filename):
        stem = Path(video_filename).stem
        audio_path = self.audio_dir / f"{stem}.wav"

        if not audio_path.exists():
            possible_names = [
                self.audio_dir / f"audio_{stem}.wav",
                self.audio_dir / f"{stem}_audio.wav",
                self.audio_dir / video_filename.replace('.mp4', '.wav').replace('.MOV', '.wav').replace('.mov', '.wav')
            ]

            for path in possible_names:
                if path.exists():
                    return path

            raise FileNotFoundError(f"Аудиофайл для {video_filename} не найден в {self.audio_dir}")

        return audio_path

    def load_audio(self, audio_path):
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)

        y, sr = librosa.load(str(audio_path), sr=self.sample_rate)
        return y, sr

    def extract_mfcc(self, y, sr, n_mfcc=13):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_min = np.min(mfcc, axis=1)
        mfcc_max = np.max(mfcc, axis=1)

        return {
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std,
            'mfcc_min': mfcc_min,
            'mfcc_max': mfcc_max
        }

    def extract_pitch(self, y, sr):
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=50, fmax=500, sr=sr
        )
        f0_clean = f0[f0 > 0]

        if len(f0_clean) > 0:
            f0_mean = np.mean(f0_clean)
            f0_std = np.std(f0_clean)
            f0_min = np.min(f0_clean)
            f0_max = np.max(f0_clean)
            voiced_ratio = np.sum(voiced_flag) / len(voiced_flag)
        else:
            f0_mean = f0_std = f0_min = f0_max = voiced_ratio = 0

        return {
            'f0_mean': f0_mean,
            'f0_std': f0_std,
            'f0_min': f0_min,
            'f0_max': f0_max,
            'voiced_ratio': voiced_ratio
        }

    def extract_energy(self, y):
        rms = librosa.feature.rms(y=y)

        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        energy_min = np.min(rms)
        energy_max = np.max(rms)

        return {
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'energy_min': energy_min,
            'energy_max': energy_max
        }

    def detect_pauses(self, y, sr, threshold_db=-30):
        pauses = librosa.effects.split(y, top_db=abs(threshold_db))

        total_duration = len(y) / sr
        speech_duration = sum([end - start for start, end in pauses]) / sr
        pause_duration = total_duration - speech_duration

        num_pauses = len(pauses) - 1 if len(pauses) > 1 else 0
        pause_ratio = pause_duration / total_duration if total_duration > 0 else 0

        return {
            'num_pauses': num_pauses,
            'pause_duration': pause_duration,
            'pause_ratio': pause_ratio,
            'speech_duration': speech_duration,
            'total_duration': total_duration
        }

    def extract_all_features(self, video_filename):
        audio_path = self.get_audio_path(video_filename)

        y, sr = self.load_audio(audio_path)

        mfcc_features = self.extract_mfcc(y, sr)
        pitch_features = self.extract_pitch(y, sr)
        energy_features = self.extract_energy(y)
        pause_features = self.detect_pauses(y, sr)

        features = {
            **mfcc_features,
            **pitch_features,
            **energy_features,
            **pause_features,
            'audio_path': str(audio_path)
        }

        return features

    def process_batch(self, video_filenames):
        all_features = []
        for video_filename in video_filenames:
            try:
                features = self.extract_all_features(video_filename)
                all_features.append(features)
            except Exception as e:
                print(f"Ошибка обработки аудио для {video_filename}: {e}")
                all_features.append(None)
        return all_features

class VideoPreprocessor:
    def __init__(self, video_dir="video_files", num_frames=10, frame_size=(224, 224)):
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.frame_size = frame_size

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_video_path(self, video_filename):
        video_path = self.video_dir / video_filename

        if not video_path.exists():
            raise FileNotFoundError(f"Видеофайл {video_filename} не найден в {self.video_dir}")

        return video_path

    def extract_frames(self, video_path, num_frames=None):
        if num_frames is None:
            num_frames = self.num_frames

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            raise ValueError(f"Видео содержит 0 кадров: {video_path}")

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        frame_paths = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

            ret, frame = cap.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            else:
                print(f"Предупреждение: не удалось прочитать кадр {idx}")

        cap.release()

        return frames

    def preprocess_frame(self, frame):
        frame_pil = Image.fromarray(frame)

        frame_tensor = self.transform(frame_pil)

        return frame_tensor

    def preprocess_video(self, video_filename, num_frames=None):
        video_path = self.get_video_path(video_filename)

        frames = self.extract_frames(video_path, num_frames)

        processed_frames = []
        for frame in frames:
            processed = self.preprocess_frame(frame)
            processed_frames.append(processed)

        video_tensor = torch.stack(processed_frames)

        return video_tensor, frames, str(video_path)

    def process_batch(self, video_filenames, num_frames=None):
        all_tensors = []
        all_frames = []
        all_paths = []

        for video_filename in video_filenames:
            try:
                tensor, frames, video_path = self.preprocess_video(video_filename, num_frames)
                all_tensors.append(tensor)
                all_frames.append(frames)
                all_paths.append(video_path)
            except Exception as e:
                print(f"Ошибка обработки {video_filename}: {e}")
                all_tensors.append(None)
                all_frames.append(None)
                all_paths.append(None)

        return all_tensors, all_frames, all_paths

class MultimodalPreprocessor:
    def __init__(self, video_dir="video_files", audio_dir="audio_files"):
        self.video_dir = Path(video_dir)
        self.audio_dir = Path(audio_dir)

        self.text_processor = TextPreprocessor()
        self.audio_processor = AudioPreprocessor(audio_dir=str(audio_dir))
        self.video_processor = VideoPreprocessor(video_dir=str(video_dir))

    def process_single_sample(self, text, video_filename):

        print(f"Обработка текста...")
        text_embedding = self.text_processor.get_embeddings(text)

        print(f"Обработка аудио...")
        audio_features = self.audio_processor.extract_all_features(video_filename)

        print(f"Обработка видео...")
        video_tensor, frames, video_path = self.video_processor.preprocess_video(video_filename)

        return {
            'text_embedding': text_embedding,
            'audio_features': audio_features,
            'video_tensor': video_tensor,
            'frames': frames,
            'video_path': video_path,
            'audio_path': audio_features.get('audio_path', '')
        }

    def process_dataset(self, df, text_column='text', video_column='video_file'):

        results = []

        for idx, row in df.iterrows():
            print(f"\nОбработка записи {idx + 1}/{len(df)}")

            text = row[text_column]
            video_filename = row[video_column]

            try:
                sample_data = self.process_single_sample(text, video_filename)
                sample_data['index'] = idx
                sample_data['original_text'] = text
                sample_data['video_filename'] = video_filename
                results.append(sample_data)

                print(f"Запись {idx + 1} успешно обработана")
            except Exception as e:
                print(f"Ошибка обработки записи {idx + 1}: {e}")
                results.append({
                    'index': idx,
                    'error': str(e),
                    'video_filename': video_filename
                })

        return results

    def save_results(self, results, output_path):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        text_embeddings = [r['text_embedding'] for r in results if 'text_embedding' in r]
        if text_embeddings:
            np.save(output_path / 'text_embeddings.npy', np.array(text_embeddings))
            print(f"Текстовые эмбеддинги сохранены: {output_path / 'text_embeddings.npy'}")

        audio_features = [r['audio_features'] for r in results if 'audio_features' in r]
        if audio_features:
            clean_features = []
            for feat in audio_features:
                clean_feat = {k: v for k, v in feat.items() if k != 'audio_path'}
                clean_features.append(clean_feat)

            audio_df = pd.DataFrame(clean_features)
            audio_df.to_csv(output_path / 'audio_features.csv', index=False)
            print(f"Аудиопризнаки сохранены: {output_path / 'audio_features.csv'}")

        video_tensors = [r['video_tensor'] for r in results if 'video_tensor' in r]
        if video_tensors:
            video_stack = torch.stack(video_tensors)
            torch.save(video_stack, output_path / 'video_tensors.pt')
            print(f"Видеотензоры сохранены: {output_path / 'video_tensors.pt'}")

        metadata = {
            'total_samples': len(results),
            'successful': sum(1 for r in results if 'text_embedding' in r),
            'failed': sum(1 for r in results if 'error' in r),
            'video_dir': str(self.video_dir),
            'audio_dir': str(self.audio_dir)
        }
        import json
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Метаданные сохранены: {output_path / 'metadata.json'}")

        file_paths = []
        for r in results:
            if 'video_filename' in r:
                file_paths.append({
                    'index': r['index'],
                    'video_filename': r.get('video_filename', ''),
                    'video_path': r.get('video_path', ''),
                    'audio_path': r.get('audio_path', ''),
                    'has_error': 'error' in r
                })

        paths_df = pd.DataFrame(file_paths)
        paths_df.to_csv(output_path / 'file_paths.csv', index=False)
        print(f"Пути к файлам сохранены: {output_path / 'file_paths.csv'}")

        return output_path

def main():

    ANNOTATION_FILE = "first_annotated_data/annotation.xlsx"
    VIDEO_DIR = "processed_russian_speech_dataset/video_with_audio"
    AUDIO_DIR = "processed_russian_speech_dataset/audio_only"
    OUTPUT_DIR = "preprocessed_data"

    video_dir = Path(VIDEO_DIR)
    audio_dir = Path(AUDIO_DIR)

    if not video_dir.exists():
        print(f"Папка с видео не найдена: {video_dir}")
        sys.exit(1)

    if not audio_dir.exists():
        print(f"Папка с аудио не найдена: {audio_dir}")
        sys.exit(1)

    print(f"Папка с видео: {video_dir}")
    print(f"Папка с аудио: {audio_dir}")

    df = pd.read_excel(ANNOTATION_FILE)
    print(f"Загружено {len(df)} записей")

    video_files_found = 0
    for video_filename in df['video_file']:
        video_path = video_dir / video_filename
        if video_path.exists():
            video_files_found += 1
        else:
            print(f"Видеофайл не найден: {video_filename}")

    print(f"Найдено {video_files_found} из {len(df)} видеофайлов")

    audio_processor = AudioPreprocessor(audio_dir=str(audio_dir))
    audio_files_found = 0

    for video_filename in df['video_file']:
        try:
            audio_path = audio_processor.get_audio_path(video_filename)
            if Path(audio_path).exists():
                audio_files_found += 1
        except FileNotFoundError:
            print(f"Аудиофайл не найден для: {video_filename}")

    print(f"Найдено {audio_files_found} из {len(df)} аудиофайлов")

    preprocessor = MultimodalPreprocessor(video_dir=VIDEO_DIR, audio_dir=AUDIO_DIR)

    results = preprocessor.process_dataset(df)

    output_path = preprocessor.save_results(results, OUTPUT_DIR)

    successful = sum(1 for r in results if 'text_embedding' in r)
    failed = sum(1 for r in results if 'error' in r)

    print(f"Всего записей: {len(results)}")
    print(f"Успешно обработано: {successful}")
    print(f"Ошибок: {failed}")
    print(f"\nРезультаты сохранены в: {output_path}")

if __name__ == "__main__":
    main()