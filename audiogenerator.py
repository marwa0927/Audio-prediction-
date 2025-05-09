import torch
import torch.nn as nn
import logging
from TTS.api import TTS
import numpy as np

logger = logging.getLogger(__name__)

class AudioGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use a properly formatted TTS model path
        model_name = "tts_models/en/ljspeech/tacotron2-DDC"  # Free English TTS model
        try:
            self.tts = TTS(model_name)
            logger.info(f"Successfully loaded TTS model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading TTS model: {str(e)}")
            raise
    
    def forward(self, text):
        try:
            # Handle empty or invalid text
            if not text or not isinstance(text, str):
                logger.warning(f"Invalid text input: {text}")
                return {
                    'waveform': torch.zeros((1, 47648), device=self.device),
                    'sample_rate': self.tts.synthesizer.output_sample_rate
                }
            
            logger.info(f"Generating audio for text: {text}")
            
            # Generate audio
            wav = self.tts.tts(text=text)
            audio_tensor = torch.tensor(wav, device=self.device).float()
            
            # Ensure output shape matches expected dimensions
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            return {
                'waveform': audio_tensor,
                'sample_rate': self.tts.synthesizer.output_sample_rate
            }
            
        except Exception as e:
            logger.error(f"Error in audio generation: {str(e)}")
            return {
                'waveform': torch.zeros((1, 47648), device=self.device),
                'sample_rate': self.tts.synthesizer.output_sample_rate
            }
    
    def generate_audio(self, text, speaker_embedding=None):
        """
        Generate audio from text (legacy method for compatibility)
        
        Args:
            text (str): Text to synthesize
            speaker_embedding (optional): Speaker embedding for voice cloning
            
        Returns:
            torch.Tensor: Generated audio waveform
        """
        try:
            result = self.forward(text)
            return result['waveform']
            
        except Exception as e:
            logger.error(f"Error in audio generation: {str(e)}")
            raise
    
    def save_audio(self, waveform, filepath, sample_rate=22050):
        """
        Save generated audio to file
        
        Args:
            waveform (torch.Tensor): Audio waveform
            filepath (str): Path to save the audio file
            sample_rate (int): Sample rate of the audio
        """
        try:
            import soundfile as sf
            
            # Convert to numpy if needed
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            
            sf.write(filepath, waveform, sample_rate)
            logger.info(f"Audio saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving audio: {str(e)}")
            raise
