import torch
import cv2
import os
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from typing import List

def save_cam_image(original_path: str, mask: np.ndarray, save_path: str) -> None:
    """元の画像とGrad-CAMのヒートマップを合成して保存する"""

    img = cv2.imread(original_path)
    if img is None: 
        return
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    background = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    mask_resized = cv2.resize(mask, (w, h))
    mask_u8 = (mask_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)
    
    vis = cv2.addWeighted(background, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, vis)

def run_cam_loop(
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
        dataset, 
        class_names: List[str], 
        save_dir: str, 
        target_layer: list, 
        device: torch.device, 
        target_rank: int, 
        dataset_type: str
        ) -> None:
    """Grad-CAMを生成し、可視化・保存する"""

    model.eval()
    cam = GradCAM(model=model, target_layers=target_layer)
    
    sub_dir = f"gradcam_rank{target_rank}"
    save_root = os.path.join(save_dir, sub_dir)
    os.makedirs(save_root, exist_ok=True)

    # クラスのインデックスを得る
    idx_p1 = class_names.index('Period_I')
    idx_p2 = class_names.index('Period_II')
    idx_p3 = class_names.index('Period_III')

    val_indices = dataloader.dataset.indices if hasattr(dataloader.dataset, 'indices') else list(range(len(dataloader.dataset)))
    curr_idx = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        
        with torch.no_grad():
            output = model(inputs)
            probs = torch.nn.functional.softmax(output, dim=1)
            topk_probs, topk_inds = torch.topk(probs, 3, dim=1)

        for i in range(inputs.size(0)):
            label_idx = labels[i].item()
            true_name = class_names[label_idx]
            
            # Period_IIのみ抽出して可視化する
            if true_name != "Period_II":
                curr_idx += 1
                continue

            # 元画像のパス
            img_idx = val_indices[curr_idx]
            if hasattr(dataset, 'imgs'):
                original_path = dataset.imgs[img_idx][0]
            elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'imgs'):
                 original_path = dataset.dataset.imgs[img_idx][0]
            else:
                curr_idx += 1
                continue

            basename = os.path.splitext(os.path.basename(original_path))[0]

            if target_rank > topk_inds.size(1):
                curr_idx += 1
                continue

            pred_idx = topk_inds[i][target_rank - 1].item()
            pred_name = class_names[pred_idx]

            prob_p1 = probs[i, idx_p1].item()
            prob_p2 = probs[i, idx_p2].item()
            prob_p3 = probs[i, idx_p3].item()
            prob_dist_str = f"[{prob_p1:.4f}, {prob_p2:.4f}, {prob_p3:.4f}]"

            log_str = f"{dataset_type},{target_rank},{basename},{true_name},{pred_name},{prob_dist_str}"
            print(log_str)

            targets = [ClassifierOutputTarget(pred_idx)]
            input_tensor = inputs[i:i+1]
            mask = cam(input_tensor=input_tensor, targets=targets)[0, :]
            
            filename = f"{basename}_True-{true_name}_Rank{target_rank}-{pred_name}.png"
            save_cam_image(original_path, mask, os.path.join(save_root, filename))
            
            curr_idx += 1