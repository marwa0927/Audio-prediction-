import torch
import torch.nn as nn
import logging
import torchaudio
import torchaudio.transforms as T

logger = logging.getLogger(__name__)

class AudioSegmentHandler(nn.Module):
    def _process_generated_segment(self, generated_audio, batch_idx, gap_length, device):
        """Process generated audio segment to match required length"""
        if generated_audio.dim() > 1:
            gen_audio = generated_audio[batch_idx]
        else:
            gen_audio = generated_audio
            
        if gen_audio.size(-1) != gap_length:
            gen_audio = torch.nn.functional.interpolate(
                gen_audio.unsqueeze(0).unsqueeze(0),
                size=gap_length,
                mode='linear',
                align_corners=False
            ).squeeze()
            
        return gen_audio.to(device)
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sample_rate = config.get('audio', {}).get('sample_rate', 16000)
        
        # Audio processing modules
        self.resample = T.Resample(
            orig_freq=self.sample_rate, 
            new_freq=self.sample_rate
        )
        
    def _process_gaps(self, gaps):
        """Convert gaps to standard format"""
        processed_gaps = []
        for batch_gaps in gaps:
            batch_processed = []
            for gap in batch_gaps:
                if isinstance(gap, (tuple, list)):
                    start, end = gap
                    batch_processed.append({'start': start, 'end': end})
                elif isinstance(gap, dict):
                    batch_processed.append(gap)
                else:
                    logger.warning(f"Skipping invalid gap format: {gap}")
            processed_gaps.append(batch_processed)
        return processed_gaps

    def _apply_crossfade_and_insert(self, original_audio, generated_audio, start_sample, end_sample, device):
        """Apply crossfade between original and generated audio"""
        gap_length = end_sample - start_sample
        crossfade_length = min(1000, gap_length // 4)
        
        if crossfade_length > 0:
            # Create fade curves
            fade_in = torch.linspace(0, 1, crossfade_length, device=device)
            fade_out = torch.linspace(1, 0, crossfade_length, device=device)
            
            # Apply crossfade at the start
            original_audio[start_sample:start_sample + crossfade_length] *= fade_out
            generated_audio[:crossfade_length] *= fade_in
            
            # Apply crossfade at the end
            original_audio[end_sample - crossfade_length:end_sample] *= fade_in
            generated_audio[-crossfade_length:] *= fade_out
        
        # Insert generated audio
        original_audio[start_sample:end_sample] = generated_audio
        
        return original_audio

    def combine_audio(self, original_audio, generated_audio, gaps):
        try:
            device = original_audio.device
            batch_size = original_audio.size(0)
            output_audio = original_audio.clone()
            
            # Ensure generated audio has correct batch size
            if isinstance(generated_audio, dict):
                generated_audio = generated_audio['waveform']
            
            # Repeat generated audio for each batch item if needed
            if generated_audio.size(0) == 1 and batch_size > 1:
                generated_audio = generated_audio.repeat(batch_size, 1)
            
            generated_audio = generated_audio.to(device)
            
            # Process each batch
            for batch_idx, batch_gaps in enumerate(gaps):
                for gap in batch_gaps:
                    start_sample, end_sample = gap
                    gap_length = end_sample - start_sample
                    
                    # Get segment of generated audio
                    gen_audio = generated_audio[batch_idx]
                    
                    # Resize if needed
                    if gen_audio.size(-1) != gap_length:
                        gen_audio = torch.nn.functional.interpolate(
                            gen_audio.unsqueeze(0).unsqueeze(0),
                            size=gap_length,
                            mode='linear',
                            align_corners=False
                        ).squeeze()
                    
                    # Apply crossfade
                    crossfade_length = min(1000, gap_length // 4)
                    if crossfade_length > 0:
                        fade_in = torch.linspace(0, 1, crossfade_length, device=device)
                        fade_out = torch.linspace(1, 0, crossfade_length, device=device)
                        
                        # Apply fades
                        output_audio[batch_idx, start_sample:start_sample + crossfade_length] *= fade_out
                        output_audio[batch_idx, end_sample - crossfade_length:end_sample] *= fade_in
                        
                        gen_audio[:crossfade_length] *= fade_in
                        gen_audio[-crossfade_length:] *= fade_out
                    
                    # Insert audio
                    output_audio[batch_idx, start_sample:end_sample] = gen_audio
            
            return output_audio.requires_grad_(True)
            
        except Exception as e:
            logger.error(f"Error in audio combination: {str(e)}")
            logger.error(f"Original audio shape: {original_audio.shape}")
            logger.error(f"Generated audio shape: {generated_audio.shape}")
            logger.error(f"Gaps: {gaps}")
            return original_audio.requires_grad_(True)
