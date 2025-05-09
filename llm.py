import torch
import torch.nn as nn
import logging
from transformers import GPT2Tokenizer, GPT2LMHeadModel

logger = logging.getLogger(__name__)

class LLMEnhancer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Add device initialization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Initialize model and move to device
        model_name = "distilgpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        
        
    def enhance_predictions(self, stt_pred, lip_features):
        try:
            # If no STT or lip reading predictions are available, return empty
            if stt_pred is None and lip_features is None:
                return {
                    'enhanced_text': "",
                    'features': None
                }
            
            # Extract STT transcription (excluding the gap)
            stt_text = stt_pred.get('transcription', [""])[0] if stt_pred else ""
            
            # Extract lip reading predictions (entire video, including the gap)
            lip_text = lip_features.get('text_predictions', [""])[0] if lip_features else ""
            
            # Create a concise prompt to fill in the missing audio gap
            prompt = (
                "Fill in the missing audio gap in the following transcription using the lip reading predictions. "
                f"STT Transcription (excluding gap): {stt_text}. "
                f"Lip Reading Prediction (entire video): {lip_text}."
            )
            
            # Tokenize the prompt and move to the appropriate device
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate enhanced text using GPT-2
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=100,
                do_sample=True,  # Enable sampling
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode the generated text and clean it up
            enhanced_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the output
            enhanced_text = enhanced_text.replace("Fill in the missing audio gap in the following transcription using the lip reading predictions.", "").strip()
            enhanced_text = enhanced_text.replace("STT Transcription (excluding gap):", "").strip()
            enhanced_text = enhanced_text.replace("Lip Reading Prediction (entire video):", "").strip()
            
            return {
                'enhanced_text': enhanced_text,
                'features': stt_pred.get('features', None) if stt_pred else None
            }
            
        except Exception as e:
            logger.error(f"Error in prediction enhancement: {str(e)}")
            # Return the original STT or lip reading text if an error occurs
            return {
                'enhanced_text': stt_text if stt_pred else lip_text,
                'features': lip_features
            }
