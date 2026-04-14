import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from typing import Tuple, Dict, List

import utils_config

def get_dataloaders(
        data_dir: str, 
        batch_size: int, 
        generator: torch.Generator, 
        num_workers: int = 16
        ) -> Tuple[Dict[str, DataLoader], 
                   Dict[str, datasets.ImageFolder], 
                   Dict[str, int], 
                   List[str]]:
    """データセットからDataLoaderを作成する"""

    # 前処理
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(utils_config.MEAN, utils_config.STD)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(utils_config.MEAN, utils_config.STD)
        ]),
    }

    # データセットの読み込み
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
    }

    # DataLoaderの作成
    dataloaders = {
        'train': DataLoader(
            image_datasets['train'], 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            generator=generator            
        ),
        'val': DataLoader(
            image_datasets['val'], 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        ),
        'train_plain': DataLoader(
            image_datasets['train'],
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_workers
        )
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    print(f"Training data size: {dataset_sizes['train']}")
    print(f"Validation data size: {dataset_sizes['val']}")
    print(f"Classes: {class_names}")
    
    return dataloaders, image_datasets, dataset_sizes, class_names