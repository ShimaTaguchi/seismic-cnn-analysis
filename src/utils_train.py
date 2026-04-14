import torch
import copy
import logging
from typing import Tuple, Dict, Optional

def train_model(
                model: torch.nn.Module, 
                criterion: torch.nn.modules.loss._Loss, 
                optimizer: torch.optim.Optimizer, num_epochs: int, 
                dataloaders: Dict[str, torch.utils.data.DataLoader], 
                dataset_sizes: Dict[str, int], device: torch.device, 
                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, 
                logger: Optional[logging.Logger] = None
                ) -> Tuple[torch.nn.Module, Dict[str, list]]:
    """モデルの学習と検証ループを実行する"""
    
    def _print(msg: str):
        if logger: logger.info(msg)
        else: print(msg)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        _print(f'Epoch {epoch+1}/{num_epochs}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            _print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # 最も良いモデルを保存する
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    _print(f'New best val Acc: {best_acc:.4f} (Loss: {best_loss:.4f})')
                    
                elif epoch_acc == best_acc and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    _print(f'Same Acc ({best_acc:.4f}), but lower Loss: {best_loss:.4f} -> Update')
    
    _print(f'Training complete. Best val Acc: {best_acc:.4f} (Loss: {best_loss:.4f})')

    # 最も良い重みを返す
    model.load_state_dict(best_model_wts)
    return model, history