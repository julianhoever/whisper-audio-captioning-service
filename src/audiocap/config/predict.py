ARCHITECTURE = dict(
    name="openai/whisper-large-v2",
)
RUNTIME = dict(
    use_fp16=True,
    device="cuda",
)
DATALOADER = dict(
    batch_size=2,
    num_workers=4,
    pin_memory=True,
)
GENERATE = dict(
    max_length=80,
    num_beams=5,
)
DATASET = dict(
    source_ds="clotho",
    task="caption",
)
