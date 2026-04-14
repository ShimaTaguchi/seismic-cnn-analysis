import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, List, Any

import utils_config
import utils_data
import utils_model
import utils_train
import utils_eval
import utils_gradcam

def setup_logger(log_path: str) -> Tuple[logging.Logger, logging.FileHandler, logging.StreamHandler]:
    """ログ出力の設定"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')

    # ファイルへの出力
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # ターミナルへの出力
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger, fh, sh

def run_gradcam_all_ranks(
    model: nn.Module, 
    dataloader: DataLoader, 
    dataset: Any, 
    class_names: List[str], 
    save_dir: str, 
    target_layer: List[nn.Module], 
    device: torch.device, 
    dataset_type: str
) -> None:
    """Rank1, 2, 3のGrad-CAMを実行して保存する"""
    
    save_subdir = os.path.join(save_dir, dataset_type) 
    os.makedirs(save_subdir, exist_ok=True)
    
    for rank in [1, 2, 3]:
        utils_gradcam.run_cam_loop(
            model=model, 
            dataloader=dataloader, 
            dataset=dataset, 
            class_names=class_names, 
            save_dir=save_subdir,
            target_layer=target_layer, 
            device=device, 
            target_rank=rank,
            dataset_type=dataset_type
        )

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    device = utils_config.get_device()
    all_results = {}

    for seed in args.seeds:
        seed_dir = os.path.join(args.out_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        
        # 初期化
        log_path = os.path.join(seed_dir, f'log_{seed}.txt')
        logger, fh, sh = setup_logger(log_path)
        
        try:
            logger.info(f"\n{'='*60}\nSEED: {seed}\n{'='*60}")
            
            # シード
            utils_config.set_seed(seed)
            g_seed = torch.Generator()
            g_seed.manual_seed(seed)

            # データの読み込み
            dataloaders, image_datasets, dataset_sizes, class_names = utils_data.get_dataloaders(
                args.data_dir, args.batch_size, g_seed
            )
            model = utils_model.initialize_model(args.num_classes, device)
            criterion = nn.CrossEntropyLoss()

            # ----------------------------------------------------
            # 1. 転移学習 (Headのみの学習)
            # ----------------------------------------------------
            logger.info("Starting Phase 1: Head Training")
            for param in model.parameters(): 
                param.requires_grad = False
            for param in model.classifier.parameters(): 
                param.requires_grad = True
                
            optimizer = optim.AdamW(model.parameters(), lr=args.lr_head, weight_decay=1e-2)
            model, history_head = utils_train.train_model(
                model, criterion, optimizer, args.epochs_head, dataloaders, dataset_sizes, device, logger=logger
            )

            # ----------------------------------------------------
            # 2. ファインチューニング (全層の学習)
            # ----------------------------------------------------
            logger.info("Starting Phase 2: Fine-Tuning")
            for param in model.parameters(): 
                param.requires_grad = True
                
            optimizer = optim.AdamW(model.parameters(), lr=args.lr_ft, weight_decay=1e-2)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_ft)
            model, history_ft = utils_train.train_model(
                model, criterion, optimizer, args.epochs_ft, dataloaders, dataset_sizes, device, scheduler=scheduler, logger=logger
            )

            # 履歴の結合
            history_head['val_acc'].extend(history_ft['val_acc'])
            history_head['train_loss'].extend(history_ft['train_loss'])
            history_head['train_acc'].extend(history_ft['train_acc'])
            history_head['val_loss'].extend(history_ft['val_loss'])

            # ----------------------------------------------------
            # 3. 評価とGrad-CAMによる可視化
            # ----------------------------------------------------
            utils_eval.plot_history(history_head, seed_dir)
            utils_eval.evaluate_model(model, dataloaders['val'], class_names, seed_dir, device, logger=logger)

            target_layer = [model.features[-1]]
            
            # Validation画像のGrad-CAM
            run_gradcam_all_ranks(
                model, dataloaders['val'], image_datasets['val'], class_names, 
                seed_dir, target_layer, device, dataset_type="val"
            )
            
            # Train画像のGrad-CAM
            train_loader_cam = dataloaders.get(
                'train_plain', 
                DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=False, num_workers=4)
            )
            run_gradcam_all_ranks(
                model, train_loader_cam, image_datasets['train'], class_names, 
                seed_dir, target_layer, device, dataset_type="train"
            )

            # 最高精度の記録
            best_val_acc = max(history_head['val_acc']) if history_head['val_acc'] else 0.0
            all_results[seed] = best_val_acc
            logger.info(f"Best Val Accuracy for seed {seed}: {best_val_acc:.4f}")

        finally:
            logger.removeHandler(fh)
            logger.removeHandler(sh)
            fh.close()
            sh.close()
            
    # 全シードの結果を出力
    print("\n--- Final Results ---")
    for seed, acc in all_results.items():
        print(f"Seed {seed}: {acc:.4f}")
    print(f"Average: {sum(all_results.values()) / len(all_results):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EfficientNet Transfer Learning & Grad-CAM Pipeline")
    parser.add_argument('--data_dir', type=str, default='dataset_STFT_gouraud224_1223', help='データセットのディレクトリ')
    parser.add_argument('--out_dir', type=str, default='output/20260305/EfficientNet_gouraud_1223', help='出力先ディレクトリ')
    parser.add_argument('--num_classes', type=int, default=3, help='分類クラス数')
    parser.add_argument('--batch_size', type=int, default=16, help='バッチサイズ')
    parser.add_argument('--epochs_head', type=int, default=20, help='Head学習のエポック数')
    parser.add_argument('--epochs_ft', type=int, default=20, help='ファインチューニングのエポック数')
    parser.add_argument('--lr_head', type=float, default=1e-3, help='Head学習の学習率')
    parser.add_argument('--lr_ft', type=float, default=1e-5, help='ファインチューニングの学習率')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], help='実行するシード値のリスト')
    
    args = parser.parse_args()
    main(args)