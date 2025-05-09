import torch
import torch.nn as nn
import logging
from pathlib import Path
from models.speech_to_text import SpeechToTextModule
from models.lip_reading import LipReadingModule
from models.llm_enhancer import LLMEnhancer
from models.audio_generator import AudioGenerator
from models.audio_handler import AudioSegmentHandler

logger = logging.getLogger(__name__)

class AudioInpaintingPipeline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setup_modules()
        
    def setup_modules(self):
        try:
            logger.info("Setting up pipeline modules...")
            
            # Initialize all modules with config
            self.stt_module = SpeechToTextModule(self.config)
            logger.info("STT module initialized")
            
            # Pass config correctly to LipReadingModule
            self.lip_module = LipReadingModule(config=self.config)  
            logger.info("Lip reading module initialized")
            
            self.llm_enhancer = LLMEnhancer(self.config)
            logger.info("LLM enhancer initialized")
            
            self.audio_generator = AudioGenerator(self.config)
            logger.info("Audio generator initialized")
            
            self.audio_handler = AudioSegmentHandler(self.config)
            logger.info("Audio handler initialized")
            
        except Exception as e:
            logger.error(f"Error in setup_modules: {str(e)}")
            raise
    def forward(self, batch):
        try:
            # Process video input
            lip_features = self.lip_module(batch['video'])
            logger.info(f"Lip reading predictions: {lip_features['text_predictions']}")
            
            # Process audio if available
            audio_features = None
            if 'audio' in batch:
                audio = batch['audio']
                if audio.dim() == 3:
                    audio = audio.squeeze(1)
                
                logger.info(f"Audio shape in pipeline: {audio.shape}")
                
                # Get audio features and transcription
                audio_features = self.stt_module(audio, batch.get('gaps', []))
                logger.info(f"STT transcriptions: {audio_features['transcription']}")
            
            # Enhance predictions
            enhanced = self.llm_enhancer.enhance_predictions(
                audio_features,
                lip_features
            )
            logger.info(f"Enhanced text: {enhanced['enhanced_text']}")
            
            # Generate audio
            generated_audio = self.audio_generator(enhanced['enhanced_text'])
            
            # Handle audio segments
            if 'audio' in batch:
                final_audio = self.audio_handler.combine_audio(
                    batch['audio'],
                    generated_audio['waveform'],
                    batch.get('gaps', [])
                )
            else:
                final_audio = generated_audio['waveform']
            
            return {
                'output_audio': final_audio,
                'stt_output': audio_features['transcription'] if audio_features else None,
                'lip_output': lip_features['text_predictions'],
                'enhanced_text': enhanced['enhanced_text']
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline forward pass: {str(e)}")
            if 'audio' in batch:
                logger.error(f"Audio shape: {batch['audio'].shape}")
            raise
