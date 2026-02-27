import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple


class MOVConverter:
    def __init__(self,
                 input_dir: str = "processed_russian_speech_dataset\mov_files",
                 output_dir: str = "processed_russian_speech_dataset\video_with_audio",
                 remove_audio: bool = False,
                 codec: str = "libx264",
                 crf: int = 23):

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.remove_audio = remove_audio
        self.codec = codec
        self.crf = crf

        # winget install ffmpeg
        if not self._check_ffmpeg():
            print("ffmpeg не найден.")
            sys.exit(1)

    def _check_ffmpeg(self) -> bool:
        try:
            subprocess.run(["ffmpeg", "-version"],
                           capture_output=True,
                           check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def find_mov_files(self) -> List[Path]:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Директория не найдена: {self.input_dir}")

        mov_files = list(self.input_dir.rglob("*.MOV")) + \
                    list(self.input_dir.rglob("*.mov"))
        return mov_files

    def convert_file(self, mov_path: Path) -> Tuple[bool, str]:
        relative_path = mov_path.relative_to(self.input_dir).parent
        output_subdir = self.output_dir / relative_path
        output_subdir.mkdir(parents=True, exist_ok=True)

        output_name = mov_path.stem + ".mp4"
        output_path = output_subdir / output_name

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(mov_path),
        ]

        if self.remove_audio:
            cmd.extend(["-an"])
        else:
            cmd.extend(["-c:a", "aac", "-b:a", "128k"])

        cmd.extend([
            "-c:v", self.codec,
            "-crf", str(self.crf),
            "-preset", "medium",
            "-movflags", "+faststart",
            str(output_path)
        ])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return True, str(output_path)
        except subprocess.CalledProcessError as e:
            error_msg = f"Ошибка конвертации {mov_path.name}:\n{e.stderr[:500]}"
            return False, error_msg
        except Exception as e:
            return False, f"Неожиданная ошибка: {str(e)}"

    def convert_all(self) -> dict:

        mov_files = self.find_mov_files()
        if not mov_files:
            print(".MOV файлы не найдены")
            return {"success": 0, "failed": 0, "errors": []}

        self.output_dir.mkdir(parents=True, exist_ok=True)

        results = {"success": 0, "failed": 0, "errors": []}
        total = len(mov_files)

        for i, mov_path in enumerate(sorted(mov_files), 1):
            print(f"\n[{i}/{total}] Конвертация: {mov_path.name}")
            success, msg = self.convert_file(mov_path)
            if success:
                print(f"Сохранено: {Path(msg).name}")
                results["success"] += 1
            else:
                print(f"Ошибка: {msg[:100]}...")
                results["failed"] += 1
                results["errors"].append((mov_path.name, msg))


        print(f"Успешно: {results['success']}")
        print(f"Ошибки:  {results['failed']}")
        print(f"Всего:  {total}")

        if results["errors"]:
            with open("conversion_errors.log", "w", encoding="utf-8") as f:
                f.write("Ошибки конвертации .MOV → .MP4\n")
                f.write("=" * 70 + "\n\n")
                for filename, error in results["errors"]:
                    f.write(f"Файл: {filename}\n")
                    f.write(f"Ошибка: {error}\n")
        return results


def main():
    converter = MOVConverter(
        input_dir="processed_russian_speech_dataset/mov_files",
        output_dir="processed_russian_speech_dataset/video_with_audio",
        remove_audio=False,
        codec="libx264",
        crf=23
    )

    try:
        results = converter.convert_all()

        if results["failed"] == 0:
            print("\nВсе файлы конвертированы!")
        else:
            print(f"\nЗавершено с ошибками: {results['failed']} из {results['success'] + results['failed']}")

    except KeyboardInterrupt:
        print("\n\nКонвертация прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()