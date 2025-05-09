import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import logging

logger = logging.getLogger(__name__)


class SpeechToTextModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_name = config.get('stt_model', "facebook/wav2vec2-base-960h")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            output_hidden_states=True
        ).to(self.device)
 
    def forward(self, audio, gaps=None):
        try:
            logger.info(f"Audio shape before processing: {audio.shape}")
            
            # Detach tensor for numpy conversion
            with torch.no_grad():
                audio_np = audio.detach().cpu().numpy()
            
            # Process each audio in batch
            features = []
            transcriptions = []
            
            for i in range(audio_np.shape[0]):
                # Process audio
                inputs = self.processor(
                    audio_np[i],
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # Get transcription
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = self.processor.batch_decode(predicted_ids)
                    transcriptions.extend(transcription)
                    
                    # Log STT output
                    logger.info(f"STT transcription for batch {i}: {transcription}")
            
            return {
                'features': torch.stack(features) if features else None,
                'transcription': transcriptions
            }
            
        except Exception as e:
            logger.error(f"Error in STT forward pass: {str(e)}")
            logger.error(f"Input audio shape: {audio.shape}, device: {audio.device}")
            raise
            
    def process_audio(self, audio, gaps=None):
        """Legacy method for compatibility"""
        return self.forward(audio, gaps)
