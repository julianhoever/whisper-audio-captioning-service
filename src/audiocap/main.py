from pathlib import Path
import tempfile
from typing import Any, Dict
from fastapi import FastAPI, UploadFile

from audiocap.audio_caption_generator import AudioCaptionGenerator

app = FastAPI()
generator = AudioCaptionGenerator(
    checkpoint=Path("models/whisper-tiny-audio-captioning")
)


@app.post("/generate_caption")
def generate_caption(file: UploadFile) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_file = Path(tmpdir) / str(file.filename)
        audio_file.write_bytes(file.file.read())
        caption = generator.generate(audio_file)
    return dict(caption=caption)
