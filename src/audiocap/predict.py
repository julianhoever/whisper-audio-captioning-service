import math
import pathlib
import csv

import peft
import torch
import torch.utils.data
import typer
import transformers
import pandas as pd

import audiocap.data
import audiocap.models
import audiocap.config.predict as config

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
    output_file: pathlib.Path = typer.Option(
        ...,
        dir_okay=False,
        file_okay=True,
        exists=False,
        writable=True,
        help="Path to the folder where the predictions will be saved",
    ),
) -> None:
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_properties(i))

    batch_size = config.DATALOADER["batch_size"]
    source_ds = config.DATASET["source_ds"]
    task = config.DATASET["task"]
    use_fp16 = config.RUNTIME.get("use_fp16", False)
    device = config.RUNTIME["device"]

    checkpoint_path = pathlib.Path(checkpoint)
    if (checkpoint_path / "adapter_config.json").exists():
        # peft_config = peft.PeftConfig.from_pretrained(checkpoint_path)
        # TODO - ugly hack, should somehow find the original model weights
        model = audiocap.WhisperForAudioCaptioning.from_pretrained(
            checkpoint_path.parent / "checkpoint-orig"
        )
        model = peft.PeftModel.from_pretrained(model, checkpoint_path)
    else:
        model = audiocap.models.WhisperForAudioCaptioning.from_pretrained(checkpoint)

    tokenizer = transformers.WhisperTokenizer.from_pretrained(
        checkpoint, language="en", task="transcribe"
    )
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(
        config.ARCHITECTURE["name"]
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
        source_ds=source_ds,
        task=task,
    )

    print(f"Found: {num_files} files")

    dtype = torch.float16 if use_fp16 else torch.float32
    model = model.to(dtype).to(device).eval()

    collator = audiocap.data.DataCollatorAudioSeq2SeqWithPadding(
        tokenizer, feature_extractor, keep_cols=("file_name",)
    )
    loader = torch.utils.data.DataLoader(
        ds, **config.DATALOADER, collate_fn=collator, drop_last=False, shuffle=False
    )

    with torch.no_grad():
        for b, batch in enumerate(loader):
            print("-" * 40)
            print(f"BATCH: {b}/{math.ceil(num_files / batch_size)}")
            preds_tokens = model.generate(
                inputs=batch["input_features"].to(dtype).to(device),
                forced_ac_decoder_ids=batch["forced_ac_decoder_ids"].to(device),
                **config.GENERATE,
            )
            preds_raw: list[str] = tokenizer.batch_decode(
                preds_tokens, skip_special_tokens=False
            )
            preds = pd.Series(
                tokenizer.batch_decode(preds_tokens, skip_special_tokens=True)
            )
            preds = preds.apply(lambda x: str(x).split(":", maxsplit=1)[1].strip())

            for file_name, pred_raw in zip(batch["file_name"], preds_raw):
                print("FILE:", file_name)
                print("PRED:", pred_raw)
                print()

            df = pd.DataFrame(
                {"file_name": batch["file_name"], "caption_predicted": preds}
            )
            df.to_csv(
                output_file,
                mode="a",
                header=not output_file.exists(),
                index=False,
                encoding="utf-8",
                quoting=csv.QUOTE_NONNUMERIC,
            )


if __name__ == "__main__":
    app()
