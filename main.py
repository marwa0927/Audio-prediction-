import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import logging
import yaml
from tqdm import tqdm
from data.dataset import AudioVisualDataset  


# Add this line at the top for memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_directory_structure():
    """Test 1: Verify project directory structure"""
    logger.info("Testing directory structure...")
    
    required_dirs = ['data', 'models', 'pipeline', 'training', 'utils', 'config']
    required_files = [
        'config/config.yaml',
        'data/dataset.py',
        'pipeline/inpainting_pipeline.py',
        'training/trainer.py',
        'utils/audio_utils.py'
    ]
    
    # Check directories
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            logger.error(f"Directory {dir_name} not found!")
            return False
        if not os.path.exists(os.path.join(dir_name, '__init__.py')):
            logger.error(f"__init__.py missing in {dir_name}!")
            return False
            
    # Check files
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} not found!")
            return False
            
    logger.info("Directory structure verified successfully!")
    return True

def test_config():
    """Test 2: Verify configuration"""
    logger.info("Testing configuration...")
    
    config_path = 'config/config.yaml'
    if not os.path.exists(config_path):
        # Create default config
        config = {
            'training': {
                'batch_size': 32,
                'num_epochs': 100,
                'learning_rate': 3e-4,
                'num_workers': 4,
                'device': 'cuda'
            },
            'model': {
                'stt_model': 'large-v3',
                'llm_model': 'meta-llama/Llama-2-70b-chat-hf',
                'tts_model': 'tts_models/multilingual/multi-dataset/your_tts'
            },
            'data': {
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1
            }
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
    # Load and verify config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    required_keys = ['training', 'model', 'data']
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing {key} in config!")
            return False
            
    logger.info("Configuration verified successfully!")
    return True

def test_dataset():
    """Test 3: Verify dataset loading"""
    logger.info("Testing dataset loading...")
    
    try:
        from data.dataset import AudioVisualDataset
        
        # Get dataset path
        dataset_path = Path("/teamspace/studios/this_studio/.cache/kagglehub/datasets/jedidiahangekouakou/grid-corpus-dataset-for-training-lipnet/versions/1/data")
        
        if not dataset_path.exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return False
            
        # Test directory contents
        logger.info("\nDataset directory contents:")
        speaker_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.endswith('_processed')]
        logger.info(f"Found {len(speaker_dirs)} speaker directories")
        
        # Test first speaker directory
        if speaker_dirs:
            first_speaker = speaker_dirs[0]
            logger.info(f"\nTesting first speaker directory: {first_speaker}")
            
            # Check video files
            video_files = list(first_speaker.glob('*.mpg'))
            logger.info(f"Found {len(video_files)} video files")
            
            # Check align directory
            align_dir = first_speaker / 'align'
            if align_dir.exists():
                align_files = list(align_dir.glob('*.align'))
                logger.info(f"Found {len(align_files)} alignment files")
            else:
                logger.error("Align directory not found!")
                return False
        
        # Create dataset
        train_dataset = AudioVisualDataset(dataset_path, split='train')
        val_dataset = AudioVisualDataset(dataset_path, split='val')
        
        logger.info(f"\nTrain dataset size: {len(train_dataset)}")
        logger.info(f"Val dataset size: {len(val_dataset)}")
        
        if len(train_dataset) == 0:
            logger.error("No samples found in dataset!")
            return False
            
        # Test loading a sample
        sample = train_dataset[0]
        logger.info("\nSample contents:")
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"{k}: shape {v.shape}")
            else:
                logger.info(f"{k}: {v}")
                
        logger.info("Dataset loading verified successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing dataset: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False



def run_all_tests():
    """Run all tests"""
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Configuration", test_config),
        ("Dataset", test_dataset),
        ("Dataloader", test_dataloader)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} Testing {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} failed with error: {str(e)}")
            results.append((test_name, False))
            
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Test Summary:")
    logger.info("="*60)
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name:.<30}{status}")
    logger.info("="*60)
    
    # Return True only if all tests passed
    return all(result for _, result in results)

def create_model(config):
    """Create and initialize the model"""
    logger.info("Initializing model...")
    try:
        from pipeline.inpainting_pipeline import AudioInpaintingPipeline
        
        # Ensure model configuration exists
        if 'model' not in config:
            config['model'] = {}
        
        # Add default values if not present
        default_config = {
            'stt_model': "facebook/wav2vec2-base-960h",
            'llm_model': "distilgpt2",
            'tts_model': "tts_models/en/ljspeech/tacotron2-DDC"
        }
        
        for key, value in default_config.items():
            if key not in config['model']:
                config['model'][key] = value
        
        # Create model instance
        model = AudioInpaintingPipeline(config['model']) 
        
        # Move model to appropriate device
        device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        logger.info(f"Model initialized on device: {device}")
        return model, device
        
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise

def create_trainer(model, train_loader, val_loader, config):
    """Create and initialize the trainer"""
    logger.info("Initializing trainer...")
    try:
        from training.trainer import Trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config['training']
        )
        logger.info("Trainer initialized successfully")
        return trainer
    except Exception as e:
        logger.error(f"Error creating trainer: {str(e)}")
        raise

def train_model(trainer, config):
    """Train the model"""
    logger.info("Starting training...")
    try:
        # Create checkpoint directory
        os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
        
        # Train the model
        trainer.train()
        
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


        
def test_dataloader():
    """Test 4: Verify dataloader"""
    logger.info("Testing dataloader...")
    
    try:
        from data.dataset import AudioVisualDataset
        
        # Get dataset path
        dataset_path = Path("/teamspace/studios/this_studio/.cache/kagglehub/datasets/jedidiahangekouakou/grid-corpus-dataset-for-training-lipnet/versions/1/data")
        
        # Create dataset and dataloader
        dataset = AudioVisualDataset(dataset_path, split='train')
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=1,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2,
            collate_fn=collate_fn  
        )
        
        # Test batch loading
        batch = next(iter(dataloader))
        logger.info("\nBatch contents:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"{k}: shape {v.shape}")
            else:
                logger.info(f"{k}: type {type(v)}")
                
        logger.info("Dataloader verified successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing dataloader: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
        
def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    # Get max lengths
    max_video_len = max(x['video'].size(0) for x in batch)
    max_audio_len = max(x['audio'].size(1) for x in batch)
    
    # Initialize lists for batch
    videos = []
    audios = []
    target_audios = []
    gaps = []
    alignments = []
    speaker_ids = []
    
    # Pad sequences
    for sample in batch:
        # Pad video
        video_len = sample['video'].size(0)
        if video_len < max_video_len:
            padding = torch.zeros(max_video_len - video_len, 96, 96, 3)
            video = torch.cat([sample['video'], padding], dim=0)
        else:
            video = sample['video']
        videos.append(video)
        
        # Pad audio (ensure [1, time] shape)
        audio = sample['audio']
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio_len = audio.size(1)
        
        if audio_len < max_audio_len:
            padding = torch.zeros(1, max_audio_len - audio_len)
            audio = torch.cat([audio, padding], dim=1)
            target_audio = torch.cat([sample['target_audio'].unsqueeze(0), padding], dim=1)
        else:
            target_audio = sample['target_audio'].unsqueeze(0)
        
        audios.append(audio)
        target_audios.append(target_audio)
        
        # Store other data
        gaps.append(sample['gaps'])
        alignments.append(sample['alignment'])
        speaker_ids.append(sample['speaker_id'])
    
    # Stack tensors
    videos = torch.stack(videos)  # [batch_size, frames, height, width, channels]
    audios = torch.stack(audios)  # [batch_size, 1, time]
    target_audios = torch.stack(target_audios)  # [batch_size, 1, time]
    
    return {
        'video': videos,
        'audio': audios.squeeze(1),  # [batch_size, time]
        'target_audio': target_audios.squeeze(1),  # [batch_size, time]
        'gaps': gaps,
        'alignment': alignments,
        'speaker_id': speaker_ids,
        'sampling_rate': batch[0]['sampling_rate']
    }
def create_dataloaders(config, dataset_path):
    """Create train and validation dataloaders"""
    try:
        # Create datasets
        train_dataset = AudioVisualDataset(
            root_dir=dataset_path,
            split='train'
        )
        
        val_dataset = AudioVisualDataset(
            root_dir=dataset_path,
            split='val'
        )
        
        # Create dataloaders with custom collate function
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=1,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2,
            collate_fn=collate_fn  
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=1,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2,
            collate_fn=collate_fn  
        )
        
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"Error creating dataloaders: {str(e)}")
        raise

def main():
    try:
        # Run all tests first
        logger.info("Running preliminary tests...")
        test_results = run_all_tests()
        
        if not test_results:
            logger.error("Preliminary tests failed. Aborting training.")
            return
            
        logger.info("All preliminary tests passed. Proceeding with training setup...")
        
        # Load configuration
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Get dataset path
        dataset_path = Path("/teamspace/studios/this_studio/.cache/kagglehub/datasets/jedidiahangekouakou/grid-corpus-dataset-for-training-lipnet/versions/1/data")
        
        # Create dataloaders with verification
        train_loader, val_loader = create_dataloaders(config,dataset_path)
        
        # Create and initialize model
        model, device = create_model(config)
        
        # Create trainer
        trainer = create_trainer(model, train_loader, val_loader, config)
        
        # Train model
        train_model(trainer, config)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
