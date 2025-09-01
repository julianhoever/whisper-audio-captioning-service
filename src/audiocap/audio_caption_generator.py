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
from audiocap import config


class AudioCaptionGenerator:
    def __init__(self, checkpoint: Path) -> None:
        self._device = config.RUNTIME.device
        self._dtype = torch.float16 if config.RUNTIME.use_fp16 else torch.float32

        self._model = cast(Model, Model.from_pretrained(checkpoint))
        self._tokenizer = cast(
            Tokenizer,
            Tokenizer.from_pretrained(checkpoint, language="en", task="transcribe"),
        )
        self._feature_extractor: FeatureExtractor = cast(
            FeatureExtractor, FeatureExtractor.from_pretrained(config.ARCHITECTURE.name)
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
            source_ds=config.DATASET.source_ds,
            task=config.DATASET.task,
        )

        if num_files != 1:
            raise RuntimeError("Parent directory should only contain the audio file.")

        loader = DataLoader(
            ds,
            **config.DATALOADER.to_dict(),
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
                    **config.GENERATE.to_dict(),
                )
                preds = self._tokenizer.batch_decode(
                    preds_tokens, skip_special_tokens=True
                )
                preds = [str(x).split(":", maxsplit=1)[1].strip() for x in preds]

                return preds[0]

        return ""
