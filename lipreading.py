import os
import sys
import numpy as np
import torch
import torch.nn as nn
import logging
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model

# Ensure LipNet paths are in the PYTHONPATH
sys.path.append(os.path.abspath('LipNet'))
sys.path.insert(0, os.path.join('LipNet', 'lipnet'))

# Import custom modules and helpers
from models.lipnet_weight_loader import LipNetWeightLoader
from lipnet.model2 import LipNet  # Original model definition (if needed)
from lipreading.helpers import text_to_labels, labels_to_text
from utils.spell import Spell

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LipReadingModule(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        
        # Default configuration parameters
        self.model_config = {
            'img_c': 3,
            'img_w': 100,  # Expected width (in pixels)
            'img_h': 50,   # Expected height (in pixels)
            'frames_n': 75,
            'absolute_max_string_len': 32,
            'output_size': 28
        }
        
        # Update configuration if provided
        if config and isinstance(config, dict):
            self.model_config.update(config)
        
        # Force CPU for TensorFlow/Keras
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Path to the weight file
        weights_path = 'LipNet/evaluation/models/overlapped-weights368.h5'
        
        # Initialize the weight loader and load the Keras model
        weight_loader = LipNetWeightLoader(weights_path)
        try:
            self.keras_model = weight_loader.load_weights_safely()
            if self.keras_model is None:
                self.keras_model = weight_loader.create_model(
                    img_c=self.model_config['img_c'],
                    img_w=self.model_config['img_w'],
                    img_h=self.model_config['img_h'],
                    frames_n=self.model_config['frames_n'],
                    absolute_max_string_len=self.model_config['absolute_max_string_len'],
                    output_size=self.model_config['output_size']
                )
                logger.info("Created new model as weight loading failed.")
            else:
                logger.info("Weights loaded successfully into Keras model.")
        except Exception as e:
            logger.error(f"Error loading weights: {str(e)}")
            self.keras_model = weight_loader.create_model(
                img_c=self.model_config['img_c'],
                img_w=self.model_config['img_w'],
                img_h=self.model_config['img_h'],
                frames_n=self.model_config['frames_n'],
                absolute_max_string_len=self.model_config['absolute_max_string_len'],
                output_size=self.model_config['output_size']
            )
            logger.info("Created new model due to weight loading error.")
        
        # Initialize spell corrector
        dictionary_path = 'LipNet/common/dictionaries/grid.txt'
        self.spell_corrector = Spell(path=dictionary_path)
    
    def forward(self, video):
        """
        Perform lip reading inference.
        """
        try:
            if isinstance(video, torch.Tensor):
                video = video.detach().cpu().numpy()
            logger.info(f"Input video shape: {video.shape}")
            
            processed_video = self._preprocess_video(video)
            logger.info(f"Processed video shape: {processed_video.shape}")
            
            if self.keras_model is None:
                logger.error("Keras model is None!")
                return {'text_predictions': [''], 'raw_prediction': None}
            
            prediction = self.keras_model.predict(processed_video)
            logger.info(f"Raw prediction shape: {prediction.shape}")
            
            decoded = self._decode_prediction(prediction)
            return {'text_predictions': decoded, 'raw_prediction': prediction}
        except Exception as e:
            logger.error(f"Error in lip reading forward pass: {str(e)}", exc_info=True)
            return {'text_predictions': [''], 'raw_prediction': None}
    
    def _preprocess_video(self, video):
   
     try:
        if len(video.shape) == 4:
            video = np.expand_dims(video, axis=0)

        batch_size = video.shape[0]
        frames_in = video.shape[1]
        target_frames = self.model_config['frames_n']
        target_width = self.model_config['img_w']  # 50
        target_height = self.model_config['img_h']  # 100

        # Corrected order: (batch, frames, height, width, channels)
        resized_video = np.zeros((batch_size, target_frames, target_height, target_width, self.model_config['img_c']), dtype=np.float32)

        for b in range(batch_size):
            for f in range(min(frames_in, target_frames)):
                frame = cv2.resize(video[b, f], (target_width, target_height), interpolation=cv2.INTER_AREA)  # Fix order here
                resized_video[b, f] = frame  # Direct assignment now works correctly

        resized_video = np.clip(resized_video / 255.0, 0, 1)
        return resized_video
     except Exception as e:
        logger.error(f"Error preprocessing video: {str(e)}", exc_info=True)
        raise


    
    def _decode_prediction(self, prediction):
        """
        Decode raw model prediction and apply spell correction.
        """
        try:
            labels = prediction[0].argmax(axis=-1)
            logger.info(f"Predicted label sequence: {labels}")
            
            text = labels_to_text(labels)
            if not text:
                text = 'unknown'
            
            corrected_text = self.spell_corrector.sentence(text)
            return [corrected_text]
        except Exception as e:
            logger.error(f"Error decoding prediction: {str(e)}", exc_info=True)
            return ['unknown']
