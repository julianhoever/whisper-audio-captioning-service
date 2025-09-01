from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader

from transformers import (
    WhisperTokenizer as Tokenizer,
    WhisperFeatureExtractor as FeatureExtractor,
)

from audiocap.data import DataCollatorAudioSeq2SeqWithPadding, load_audios_for_predition
from audiocap.models import WhisperForAudioCaptioning as Model


class AudioCaptionGenerator:
    def __init__(
        self,
        checkpoint: Path,
        architecture: str = "openai/whisper-tiny",
        use_fp16: bool = False,
        device: str = "cpu",
        generate_max_length: int = 200,
        generate_num_beams: int = 5,
    ) -> None:
        self._device = device
        self._dtype = torch.float16 if use_fp16 else torch.float32
        self._generate_max_length = generate_max_length
        self._generate_num_beams = generate_num_beams

        self._model = cast(Model, Model.from_pretrained(checkpoint))
        self._tokenizer = cast(
            Tokenizer,
            Tokenizer.from_pretrained(checkpoint, language="en", task="transcribe"),
        )
        self._feature_extractor: FeatureExtractor = cast(
            FeatureExtractor, FeatureExtractor.from_pretrained(architecture)
        )
        self._collator = DataCollatorAudioSeq2SeqWithPadding(
            self._tokenizer, self._feature_extractor, keep_cols=("file_name",)
        )

        self._model = self._model.to(self._dtype).to(self._device).eval()

    def generate(self, audio_file: Path) -> str:
        ds, num_files = load_audios_for_predition(
            src=audio_file.parent,
            tokenizer=self._tokenizer,
            feature_extractor=self._feature_extractor,
            recursive=False,
            take_n=None,
            source_ds="clotho",
            task="caption",
        )

        if num_files != 1:
            raise RuntimeError("Parent directory should only contain the audio file.")

        loader = DataLoader(
            ds,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            collate_fn=self._collator,
            drop_last=False,
            shuffle=False,
        )

        with torch.no_grad():
            for batch in loader:
                preds_tokens = self._model.generate(
                    inputs=batch["input_features"].to(self._dtype).to(self._device),
                    forced_ac_decoder_ids=batch["forced_ac_decoder_ids"].to(
                        self._device
                    ),
                    max_length=self._generate_max_length,
                    num_beams=self._generate_num_beams,
                )
                preds = self._tokenizer.batch_decode(
                    preds_tokens, skip_special_tokens=True
                )
                preds = [str(x).split(":", maxsplit=1)[1].strip() for x in preds]

                return preds[0]

        return ""
