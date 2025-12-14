from dataclasses import dataclass
from torch.utils.data.dataloader import Dataloader

@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str