import os
import sys
import shutil
import subprocess
import moviepy.editor as mpy
from pathlib import Path
from typing import Dict, List, Tuple
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio

#pip install moviepy==1.0.3

class DatasetProcessor:
    def __init__(self, repo_url: str = "https://github.com/avenaki/speech-recognition-dataset.git",
                 local_dir: str = "speech_dataset_raw",
                 output_dir: str = "processed_dataset"):
        self.repo_url = repo_url
        self.local_dir = Path(local_dir)
        self.output_dir = Path(output_dir)

        self.folders = {
            "video_with_audio": self.output_dir / "video_with_audio",
            "video_only": self.output_dir / "video_only",
            "audio_only": self.output_dir / "audio_only",
            "mov_files": self.output_dir / "mov_files",
            "paired_wav": self.output_dir / "paired_wav",
            "unpaired_wav": self.output_dir / "unpaired_wav",
            "unknown": self.output_dir / "unknown"
        }

    def clone_repository(self):
        if self.local_dir.exists():
            print(f"Репозиторий уже в {self.local_dir}")
            return

        print(f"Клонирование")
        try:
            subprocess.run(
                ["git", "clone", self.repo_url, str(self.local_dir)],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"Репозиторий клонирован в {self.local_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Ошибка {e.stderr}")
            sys.exit(1)
        except FileNotFoundError:
            print("Git не найден в PATH")
            sys.exit(1)

    def create_output_structure(self):
        self.output_dir.mkdir(exist_ok=True)
        for folder in self.folders.values():
            folder.mkdir(exist_ok=True, parents=True)

    def analyze_file(self, filepath: Path) -> Dict:
        result = {
            "path": filepath,
            "stem": filepath.stem,
            "suffix": filepath.suffix.lower(),
            "has_video": False,
            "has_audio": False,
            "duration": 0.0,
            "error": None
        }

        try:
            if filepath.suffix.lower() in [".mp4", ".mov"]:
                clip = mpy.VideoFileClip(str(filepath))
                result["has_video"] = clip.w is not None and clip.h is not None
                result["has_audio"] = clip.audio is not None
                result["duration"] = clip.duration
                clip.close()

            elif filepath.suffix.lower() == ".wav":
                audio = mpy.AudioFileClip(str(filepath))
                result["has_audio"] = True
                result["duration"] = audio.duration
                audio.close()

        except Exception as e:
            result["error"] = str(e)
            print(f"Предупреждение {filepath.name}: {e}")

        return result

    def classify_and_move(self, file_info: Dict):
        src_path = file_info["path"]
        stem = file_info["stem"]
        suffix = file_info["suffix"]

        if suffix == ".mov":
            target_folder = self.folders["mov_files"]
            target_name = f"{stem}.mov"

        elif suffix == ".wav":
            has_pair = False
            for ext in [".mp4", ".MOV", ".mov"]:
                pair_path = src_path.parent / f"{stem}{ext}"
                if pair_path.exists():
                    has_pair = True
                    break

            target_folder = self.folders["paired_wav"] if has_pair else self.folders["unpaired_wav"]
            target_name = f"{stem}.wav"

        elif suffix == ".mp4":
            if file_info["has_video"] and file_info["has_audio"]:
                target_folder = self.folders["video_with_audio"]
                target_name = f"{stem}.mp4"
            elif file_info["has_video"] and not file_info["has_audio"]:
                target_folder = self.folders["video_only"]
                target_name = f"{stem}.mp4"
            elif not file_info["has_video"] and file_info["has_audio"]:
                target_folder = self.folders["audio_only"]
                target_name = f"{stem}.mp4"
            else:
                target_folder = self.folders["unknown"]
                target_name = f"{stem}.mp4"

        else:
            target_folder = self.folders["unknown"]
            target_name = src_path.name

        target_path = target_folder / target_name

        counter = 1
        while target_path.exists():
            target_path = target_folder / f"{stem}_{counter}{suffix}"
            counter += 1

        try:
            shutil.copy2(src_path, target_path)
            print(f"  → {target_folder.name}/{target_path.name}")
        except Exception as e:
            print(f"✗ Ошибка копирования {src_path.name}: {e}")

    def process_dataset(self):

        self.clone_repository()
        self.create_output_structure()
        media_files = []
        for ext in ["*.mp4", "*.MOV", "*.mov", "*.wav"]:
            media_files.extend(self.local_dir.rglob(ext))

        print(f"Найдено файлов: {len(media_files)}")

        if not media_files:
            print("Медиафайлы не найдены")
            return

        file_infos = []

        for i, filepath in enumerate(sorted(media_files), 1):
            print(f"[{i}/{len(media_files)}] Анализ: {filepath.name}", end="\r")
            info = self.analyze_file(filepath)
            file_infos.append(info)


        for info in file_infos:
            if info["error"]:
                print(f"{info['path'].name} — пропущен из-за ошибки")
                continue
            self.classify_and_move(info)


        total = 0
        for name, folder in self.folders.items():
            count = len(list(folder.glob("*")))
            total += count
            print(f"{name:20s}: {count:3d} файлов")

        print(f"Всего : {total} файлов")
        print(f"\nСтруктура в: {self.output_dir.resolve()}")

        self.generate_report(file_infos)

    def generate_report(self, file_infos: List[Dict]):
        report_path = self.output_dir / "DATASET_REPORT.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Источник: {self.repo_url}\n")
            f.write(f"Всего файлов: {len(file_infos)}\n\n")

            stats = {
                "video_with_audio": 0,
                "video_only": 0,
                "audio_only": 0,
                "mov_files": 0,
                "paired_wav": 0,
                "unpaired_wav": 0,
                "errors": 0
            }

            for info in file_infos:
                if info["error"]:
                    stats["errors"] += 1
                    continue

                if info["suffix"] == ".mov":
                    stats["mov_files"] += 1
                elif info["suffix"] == ".wav":
                    has_pair = any(
                        (info["path"].parent / f"{info['stem']}{ext}").exists()
                        for ext in [".mp4", ".MOV", ".mov"]
                    )
                    stats["paired_wav" if has_pair else "unpaired_wav"] += 1
                elif info["suffix"] == ".mp4":
                    if info["has_video"] and info["has_audio"]:
                        stats["video_with_audio"] += 1
                    elif info["has_video"]:
                        stats["video_only"] += 1
                    elif info["has_audio"]:
                        stats["audio_only"] += 1

            for key, value in stats.items():
                f.write(f"{key:20s}: {value:4d}\n")

        print(f"\nОтчёт: {report_path}")


def main():
    processor = DatasetProcessor(
        repo_url="https://github.com/avenaki/speech-recognition-dataset.git",
        local_dir="speech_dataset_raw",
        output_dir="processed_russian_speech_dataset"
    )

    try:
        processor.process_dataset()
    except KeyboardInterrupt:
        print("\n\nОбработка прервана")
        sys.exit(1)
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()