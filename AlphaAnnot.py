import os
import re
import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


class VideoAnnotator:
    def __init__(self,
                 video_dir: str = "input_videos",
                 output_dir: str = "annotated_data"):
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.annotation_columns = [
            "text",
            "context",
            "formal_text_tone",
            "prosody_pitch",
            "prosody_tempo",
            "prosody_variability",
            "prosody_pauses",
            "prosody_timbre",
            "visual_face_visible",
            "visual_quality",
            "visual_microexpressions",
            "visual_duration",
            "visual_timing",
            "true_tone",
            "mismatch_type",
            "confidence",
            "comments",
            "video_file"
        ]

    def rename_videos(self) -> List[str]:
        video_files = self._get_video_files()
        renamed_files = []

        for i, video_path in enumerate(sorted(video_files), 1):
            original_name = video_path.name

            new_name = f"video-{i}{video_path.suffix}"
            new_path = video_path.parent / new_name

            video_path.rename(new_path)
            renamed_files.append((original_name, str(new_path)))

            print(f"  {original_name} → {new_name}")

        return renamed_files

    def _get_video_files(self) -> List[Path]:
        video_exts = [".mp4", ".avi", ".mkv"]
        video_files = []
        for ext in video_exts:
            video_files.extend(self.video_dir.glob(f"*{ext}"))
        video_files.sort(key=lambda x: x.name.lower())
        if not video_files:
            raise FileNotFoundError(f"В папке {self.video_dir} не найдено видеофайлов")
        return video_files

    def create_annotation_template(self, renamed_files: List[tuple]) -> str:

        df = pd.DataFrame(columns=self.annotation_columns)

        for i, (original_name, new_name) in enumerate(renamed_files, 1):
            row = {
                "text": "",
                "context": "",
                "formal_text_tone": "",
                "prosody_pitch": "",
                "prosody_tempo": "",
                "prosody_variability": "",
                "prosody_pauses": "",
                "prosody_timbre": "",
                "visual_face_visible": "",
                "visual_quality": "",
                "visual_microexpressions": "",
                "visual_duration": "",
                "visual_timing": "",
                "true_tone": "",
                "mismatch_type": "",
                "confidence": "",
                "comments": "",
                "video_file": Path(new_name).name
            }
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        excel_path = self.output_dir / "annotation.xlsx"
        df.to_excel(excel_path, index=False)

        print(f"  Строк: {len(df)}")
        print(f"  Столбцов: {len(df.columns)}")

        return str(excel_path)

    def process(self) -> Dict[str, Any]:

        try:
            renamed_files = self.rename_videos()

            excel_path = self.create_annotation_template(renamed_files)


            return {
                "renamed_files": renamed_files,
                "excel_path": excel_path
            }

        except Exception as e:
            print(f"\nКритическая ошибка: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}


def main():
    processor = VideoAnnotator(
        video_dir="processed_russian_speech_dataset/video_with_audio",
        output_dir="first_annotated_data"
    )

    try:
        results = processor.process()
        if "error" not in results:
            print("\nОбработка завершена")
    except KeyboardInterrupt:
        print("\n\nОбработка прервана пользователем")
        sys.exit(1)


if __name__ == "__main__":
    main()