import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple
import pandas as pd


class AudioExtractor:
    def __init__(self,
                 video_dir: str = "input_videos",
                 audio_dir: str = "extracted_audio",
                 sample_rate: int = 16000,
                 channels: int = 1):

        self.video_dir = Path(video_dir)
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.channels = channels

        self.audio_dir.mkdir(exist_ok=True, parents=True)

        if not self._check_ffmpeg():
            print("ffmpeg не найден.")
            print("   Windows: https://ffmpeg.org/download.html")
            sys.exit(1)

    def _check_ffmpeg(self) -> bool:
        try:
            subprocess.run(["ffmpeg", "-version"],
                           capture_output=True,
                           check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _get_video_files(self) -> List[Path]:
        video_exts = [".mp4", ".MOV", ".mov", ".avi", ".mkv", ".flv"]
        video_files = []

        for ext in video_exts:
            video_files.extend(self.video_dir.glob(f"*{ext}"))

        video_files.sort(key=lambda x: x.name.lower())

        return video_files

    def extract_audio(self, video_path: Path) -> Tuple[bool, str]:
        audio_name = f"{video_path.stem}.wav"
        audio_path = self.audio_dir / audio_name

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", str(self.channels),
            "-ar", str(self.sample_rate),
            "-acodec", "pcm_s16le",
            "-f", "wav",
            str(audio_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return True, str(audio_path)
        except subprocess.CalledProcessError as e:
            print(f"Ошибка извлечения аудио {video_path.name}:")
            print(f"    {e.stderr[:200]}")
            return False, ""
        except Exception as e:
            print(f"Неожиданная ошибка при обработке {video_path.name}: {e}")
            return False, ""

    def extract_all(self) -> dict:
        video_files = self._get_video_files()

        if not video_files:
            return {"success": 0, "failed": 0, "errors": []}

        print(f"Найдено видеофайлов: {len(video_files)}")
        print(f"Выходная папка: {self.audio_dir.resolve()}")
        print(f"Параметры аудио: {self.sample_rate} Гц, {self.channels} канал")

        results = {"success": 0, "failed": 0, "errors": []}

        for i, video_path in enumerate(video_files, 1):
            print(f"[{i}/{len(video_files)}] Обработка: {video_path.name}")

            success, audio_path = self.extract_audio(video_path)

            if success:
                print(f"Сохранено: {Path(audio_path).name}")
                results["success"] += 1
            else:
                print(f"Не удалось извлечь аудио")
                results["failed"] += 1
                results["errors"].append(video_path.name)

        print(f"Успешно извлечено: {results['success']}")
        print(f"Ошибок: {results['failed']}")
        print(f"Всего обработано: {results['success'] + results['failed']}")

        if results["errors"]:
            report_path = self.audio_dir / "extraction_errors.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("Файлы, которые не удалось обработать:\n")
                f.write("=" * 50 + "\n\n")
                for filename in results["errors"]:
                    f.write(f"{filename}\n")
            print(f"\nОтчёт об ошибках сохранён: {report_path}")

        return results

    def update_excel_with_audio_paths(self, excel_path: str, output_excel: str = None):
        try:
            df = pd.read_excel(excel_path)

            if 'audio_file' not in df.columns:
                df['audio_file'] = None

            for idx, row in df.iterrows():
                video_file = row.get('video_file')
                if video_file:
                    audio_file = self.audio_dir / f"{Path(video_file).stem}.wav"
                    if audio_file.exists():
                        df.at[idx, 'audio_file'] = str(audio_file)

            if output_excel is None:
                output_excel = excel_path

            df.to_excel(output_excel, index=False)
            print(f"\nФайл аннотации обновлён: {output_excel}")
            print(f"Добавлено путей к аудиофайлам: {df['audio_file'].notna().sum()}")

        except Exception as e:
            print(f"Ошибка при обновлении Excel: {e}")


def main():
    VIDEO_DIR = "processed_russian_speech_dataset/video_with_audio"
    AUDIO_DIR = "processed_russian_speech_dataset/audio_only"
    EXCEL_FILE = "annotation-ver1.xlsx"
    SAMPLE_RATE = 16000
    CHANNELS = 1

    extractor = AudioExtractor(
        video_dir=VIDEO_DIR,
        audio_dir=AUDIO_DIR,
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS
    )

    try:
        results = extractor.extract_all()

        if os.path.exists(EXCEL_FILE):
            extractor.update_excel_with_audio_paths(EXCEL_FILE)

        if results["failed"] == 0:
            print("\nВсе аудиофайлы успешно извлечены!")
        else:
            print(f"\nЗавершено с ошибками: {results['failed']} из {results['success'] + results['failed']}")

        print(f"\nАудиофайлы сохранены в: {AUDIO_DIR}")

    except KeyboardInterrupt:
        print("\n\nОбработка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()