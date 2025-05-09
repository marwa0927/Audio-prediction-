from dataclasses import dataclass
from typing import Optional
import yaml
from pathlib import Path

@dataclass
class ModelConfig:
    stt_model: str = "large-v3"
    llm_model: str = "meta-llama/Llama-2-70b-chat-hf"
    tts_model: str = "tts_models/multilingual/multi-dataset/your_tts"

@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    gradient_clip_val: float = 1.0
    device: str = "cuda"
    num_workers: int = 4
    checkpoint_dir: str = "checkpoints"

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    hop_length: int = 160
    win_length: int = 400
    n_fft: int = 512
    n_mels: int = 80

@dataclass
class VideoConfig:
    frame_rate: int = 25
    image_size: int = 96
    sequence_length: int = 75

@dataclass
class DataConfig:
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    dataset_path: str = "/teamspace/studios/this_studio/.cache/kagglehub/datasets/jedidiahangekouakou/grid-corpus-dataset-for-training-lipnet/versions/1/data"

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    audio: AudioConfig = AudioConfig()
    video: VideoConfig = VideoConfig()
    data: DataConfig = DataConfig()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            audio=AudioConfig(**config_dict.get('audio', {})),
            video=VideoConfig(**config_dict.get('video', {})),
            data=DataConfig(**config_dict.get('data', {}))
        )
    
    def save(self, yaml_path: str):
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'audio': self.audio.__dict__,
            'video': self.video.__dict__,
            'data': self.data.__dict__
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
