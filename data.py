import torch
from torch.utils.data import Dataset
import logging
from pathlib import Path
import cv2
import numpy as np
import os
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)

class AudioVisualDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.sample_rate = 16000
        
        # Get all speaker directories
        self.speaker_dirs = [d for d in self.root_dir.iterdir() 
                           if d.is_dir() and d.name.endswith('_processed')]
        
        # Collect all video paths
        self.video_paths = []
        for speaker_dir in self.speaker_dirs:
            video_files = list(speaker_dir.glob('*.mpg'))
            self.video_paths.extend(video_files)
        
        # Split the dataset
        total_videos = len(self.video_paths)
        if split == 'train':
            self.video_paths = self.video_paths[:int(0.8 * total_videos)]
        elif split == 'val':
            self.video_paths = self.video_paths[int(0.8 * total_videos):int(0.9 * total_videos)]
        else:  # test
            self.video_paths = self.video_paths[int(0.9 * total_videos):]
            
        logger.info(f"Found {len(self.video_paths)} videos for {split}")
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.video_paths)
    
    def load_video(self, video_path):
        """Load and preprocess video"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert BGR to RGB and resize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (100, 50))  # Width=100, Height=50
                
                # Normalization that matches LipNet's training
                frame = frame.astype(np.float32) / 127.5 - 1.0  # [-1, 1] range
                frames.append(frame)
            
            cap.release()
            
            # Frame count handling
            if len(frames) > 75:
                frames = frames[:75]
            elif len(frames) < 75:
                frames += [frames[-1]] * (75 - len(frames))
            
            return torch.from_numpy(np.array(frames)) 
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {str(e)}")
            # Return zero tensor with correct dimensions
            return torch.zeros(75, 50, 100, 3)
    
    def load_alignment(self, align_path):
        """Load alignment file"""
        try:
            if os.path.exists(align_path):
                with open(align_path, 'r') as f:
                    return f.read().strip()
            return ""
        except Exception as e:
            logger.error(f"Error loading alignment {align_path}: {str(e)}")
            return ""
    
    def create_gaps(self, audio):
        """Create random gaps in audio for training"""
        try:
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            # Parameters for gaps
            min_gaps = 1
            max_gaps = 3
            min_gap_len = int(0.1 * self.sample_rate)  # 100ms
            max_gap_len = int(0.5 * self.sample_rate)  # 500ms
            
            # Create copy of audio
            gapped_audio = audio.clone()
            
            # Number of gaps
            num_gaps = torch.randint(min_gaps, max_gaps + 1, (1,)).item()
            gaps = []
            
            for _ in range(num_gaps):
                # Random gap length
                gap_len = torch.randint(min_gap_len, max_gap_len + 1, (1,)).item()
                
                # Random start position
                max_start = audio.size(1) - gap_len
                if max_start <= 0:
                    continue
                    
                gap_start = torch.randint(0, max_start + 1, (1,)).item()
                gap_end = gap_start + gap_len
                
                # Apply gap
                gapped_audio[..., gap_start:gap_end] = 0
                gaps.append((gap_start, gap_end))
            
            return gapped_audio, gaps
            
        except Exception as e:
            logger.error(f"Error creating gaps: {str(e)}")
            return audio, []

    def load_audio(self, video_path):
        """Extract audio from video file"""
        try:
            import av
            import torchaudio.transforms as T
            
            # Open the video file
            container = av.open(str(video_path))
            
            # Get audio stream
            audio = container.streams.audio[0]
            
            # Read all audio samples
            samples = []
            for frame in container.decode(audio=0):
                samples.append(frame.to_ndarray())
            
            # Concatenate samples
            audio_array = np.concatenate(samples, axis=1)  # Concatenate along time dimension
            
            # Convert to tensor
            waveform = torch.from_numpy(audio_array).float()
            
            # Ensure shape is [1, time]
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 2:
                if waveform.size(0) > waveform.size(1):
                    waveform = waveform.t()
                if waveform.size(0) > 1:  # If stereo
                    waveform = waveform.mean(0, keepdim=True)  # Convert to mono
            
            # Resample if needed
            if audio.rate != self.sample_rate:
                resampler = T.Resample(
                    orig_freq=audio.rate,
                    new_freq=self.sample_rate
                )
                waveform = resampler(waveform)
            
            # Normalize
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()
            
            # Create target audio (without gaps)
            target_audio = waveform.clone()
            
            # Create gaps in input audio if needed
            if self.split == 'train':
                waveform, gaps = self.create_gaps(waveform)
            else:
                gaps = []
            
            return {
                'audio': waveform,  # Shape: [1, time]
                'target_audio': target_audio,  # Shape: [1, time]
                'sampling_rate': self.sample_rate,
                'gaps': gaps
            }
            
        except Exception as e:
            logger.error(f"Error extracting audio from {video_path}: {str(e)}")
            dummy_audio = torch.zeros(1, self.sample_rate)  # [1, time]
            return {
                'audio': dummy_audio,
                'target_audio': dummy_audio.clone(),
                'sampling_rate': self.sample_rate,
                'gaps': []
            }

    def __getitem__(self, idx):
        try:
            video_path = self.video_paths[idx]
            
            # Load data
            video = self.load_video(video_path)  # [frames, height, width, channels]
            audio_data = self.load_audio(video_path)  # {'audio': [1, time], ...}
            alignment = self.load_alignment(video_path.with_suffix('.align'))
            
            # Verify shapes
            if video.dim() != 4:
                raise ValueError(f"Invalid video shape: {video.shape}")
            if audio_data['audio'].dim() != 2:  # Should be [1, time]
                raise ValueError(f"Invalid audio shape: {audio_data['audio'].shape}")
            
            return {
                'video': video,  # [frames, height, width, channels]
                'audio': audio_data['audio'],  # [1, time]
                'target_audio': audio_data['target_audio'],  # [1, time]
                'sampling_rate': audio_data['sampling_rate'],
                'gaps': audio_data['gaps'],
                'alignment': alignment,
                'speaker_id': video_path.parent.name
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            raise

 
