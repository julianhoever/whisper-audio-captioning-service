import os
from pathlib import Path
import tempfile
from typing import Any, Dict
from fastapi import FastAPI, UploadFile

from audiocap.audio_caption_generator import AudioCaptionGenerator

app = FastAPI()
generator = AudioCaptionGenerator(
    checkpoint=Path(
        os.environ.get("AC_CHECKPOINT", "models/whisper-tiny-audio-captioning")
    ),
    architecture=os.environ.get("AC_ARCHITECTURE", "openai/whisper-tiny"),
    use_fp16="AC_USE_FP16" in os.environ,
    device=os.environ.get("AC_DEVICE", "cpu"),
    generate_max_length=int(os.environ.get("AC_GENERATE_MAX_LENGTH", 200)),
    generate_num_beams=int(os.environ.get("AC_GENERATE_NUM_BEAMS", 5)),
)


@app.post("/")
def root() -> Dict[str, Any]:
    return dict()


@app.post("/generate_caption")
def generate_caption(file: UploadFile) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_file = Path(tmpdir) / str(file.filename)
        audio_file.write_bytes(file.file.read())
        caption = generator.generate(audio_file)
    return dict(caption=caption)
