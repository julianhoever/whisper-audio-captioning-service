import math
import pathlib

import torch
import torch.utils.data
import typer
import transformers

import audiocap.data
import audiocap.models
from audiocap import config

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    checkpoint: str = typer.Option(
        ...,
        dir_okay=True,
        file_okay=True,
        readable=True,
        help="Path to the checkpoint file",
    ),
    data: pathlib.Path = typer.Option(
        ...,
        dir_okay=True,
        file_okay=True,
        readable=True,
        help="Path to the file / folder with the audio files",
    ),
) -> None:
    device = config.RUNTIME.device

    model = audiocap.models.WhisperForAudioCaptioning.from_pretrained(checkpoint)
    tokenizer = transformers.WhisperTokenizer.from_pretrained(
        checkpoint, language="en", task="transcribe"
    )
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(
        config.ARCHITECTURE.name
    )

    # make mypy happy
    assert isinstance(tokenizer, transformers.WhisperTokenizer)
    assert isinstance(feature_extractor, transformers.WhisperFeatureExtractor)

    ds, num_files = audiocap.data.load_audios_for_predition(
        src=data,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        recursive=False,
        take_n=None,
        source_ds=config.DATASET.source_ds,
        task=config.DATASET.task,
    )

    print(f"Found: {num_files} files")

    dtype = torch.float16 if config.RUNTIME.use_fp16 else torch.float32
    model = model.to(dtype).to(device).eval()

    collator = audiocap.data.DataCollatorAudioSeq2SeqWithPadding(
        tokenizer, feature_extractor, keep_cols=("file_name",)
    )
    loader = torch.utils.data.DataLoader(
        ds,
        **config.DATALOADER.to_dict(),
        collate_fn=collator,
        drop_last=False,
        shuffle=False,
    )

    with torch.no_grad():
        for b, batch in enumerate(loader):
            print("-" * 40)
            print(f"BATCH: {b}/{math.ceil(num_files / config.DATALOADER.batch_size)}")
            preds_tokens = model.generate(
                inputs=batch["input_features"].to(dtype).to(device),
                forced_ac_decoder_ids=batch["forced_ac_decoder_ids"].to(device),
                **config.GENERATE.to_dict(),
            )
            preds = tokenizer.batch_decode(preds_tokens, skip_special_tokens=True)
            preds = [str(x).split(":", maxsplit=1)[1].strip() for x in preds]

            for file_name, pred in zip(batch["file_name"], preds):
                print("FILE:", file_name)
                print("PRED:", pred)
                print()


if __name__ == "__main__":
    app()
