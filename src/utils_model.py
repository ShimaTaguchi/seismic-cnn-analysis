import torch
import torch.nn as nn
from torchvision import models

def initialize_model(num_classes: int, device: torch.device) -> nn.Module:
    """EfficientNet-B0を初期化し、転移学習用にClassifier層を置き換える"""

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # 元のモデルの全結合層に入ってくる特徴量の数
    num_ftrs = model.classifier[1].in_features
    
    # クラス数に合わせてClassifier層を書き換え
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(num_ftrs, num_classes)
    )
    return model.to(device)