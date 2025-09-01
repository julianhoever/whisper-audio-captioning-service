from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class _DictConvertableDataclass:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Architecture(_DictConvertableDataclass):
    name: str


@dataclass
class Runtime(_DictConvertableDataclass):
    use_fp16: bool
    device: str


@dataclass
class DataLoader(_DictConvertableDataclass):
    batch_size: int
    num_workers: int
    pin_memory: bool


@dataclass
class Generate(_DictConvertableDataclass):
    max_length: int
    num_beams: int


@dataclass
class Dataset(_DictConvertableDataclass):
    source_ds: str
    task: str


ARCHITECTURE = Architecture(
    name="openai/whisper-tiny",
)
RUNTIME = Runtime(
    use_fp16=False,
    device="cpu",
)
DATALOADER = DataLoader(
    batch_size=2,
    num_workers=4,
    pin_memory=True,
)
GENERATE = Generate(
    max_length=80,
    num_beams=5,
)
DATASET = Dataset(
    source_ds="clotho",
    task="caption",
)
