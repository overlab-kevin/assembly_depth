# train.py

import os
import yaml
import argparse
import shutil
import torch
import segmentation_models_pytorch as smp

# Import modules from the src directory
from src.data_loader import ModelPointDataset
from src.model import UnetReg
from src.trainer import TrainEpoch, ValidationEpoch
from src.utils import get_assembly_aux_params

def main(config_path, experiment_name, resume=False):
    # --- Load Configuration ---
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # --- Setup Output Directory ---
    output_dir = os.path.join('runs', experiment_name)
    if os.path.exists(output_dir) and not resume:
        raise ValueError(f"Output directory '{output_dir}' already exists. Use --resume to continue training.")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        # Copy the config file and a version identifier to the output directory for reproducibility
        shutil.copy(config_path, os.path.join(output_dir, "config.yaml"))

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Setup DataLoaders ---
    train_dataset = ModelPointDataset(
        datapath=cfg['DATASET']['PATH'],
        phase='train',
        width=cfg['MODEL']['IMG_SIZE'][1],
        height=cfg['MODEL']['IMG_SIZE'][0],
        use_monocular_depth=cfg['MODEL']['USE_MONOCULAR_DEPTH'],
        epoch_length=cfg['DATASET']['EPOCH_LENGTH']
    )

    val_dataset = ModelPointDataset(
        datapath=cfg['DATASET']['PATH'],
        phase='val',
        width=cfg['MODEL']['IMG_SIZE'][1],
        height=cfg['MODEL']['IMG_SIZE'][0],
        use_monocular_depth=cfg['MODEL']['USE_MONOCULAR_DEPTH'],
        epoch_length=1000  # A smaller length for faster validation
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg['DATASET']['BATCH_SIZE'],
        shuffle=True,
        num_workers=cfg['DATASET']['WORKERS']
    )

    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg['DATASET']['BATCH_SIZE'],
        shuffle=False,
        num_workers=cfg['DATASET']['WORKERS']
    )

    # --- Setup Model ---
    aux_params = None
    if cfg['MODEL']['USE_ASSEMBLY_CLASSIFIER']:
        num_parts = len(train_dataset.data_interface.get_mesh_names())
        aux_params = get_assembly_aux_params(num_parts)

    # In the paper, monocular depth estimates are single-channel grayscale images.
    in_channels = 1 if cfg['MODEL']['USE_MONOCULAR_DEPTH'] else 3

    model = UnetReg(
        encoder_name=cfg['MODEL']['ENCODER_NAME'],
        in_channels=in_channels,
        classes=1,  # Binary segmentation mask
        aux_params=aux_params
    )

    model.to(device)

    # --- Setup Optimizer, Loss, and Metrics ---
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['TRAINING']['LEARNING_RATE'])
    
    # The paper uses a combination of losses, handled inside the trainer.
    # We use MSE here for the segmentation head as in the original code.
    seg_loss = smp.utils.losses.MSELoss()
    
    metrics = [
        smp.utils.metrics.Fscore(),
        smp.utils.metrics.Accuracy(),
    ]

    # --- Setup Trainer ---
    train_epoch = TrainEpoch(
        model,
        loss=seg_loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
        train_assembly=cfg['MODEL']['USE_ASSEMBLY_CLASSIFIER']
    )

    valid_epoch = ValidationEpoch(
        model,
        loss=seg_loss,
        metrics=metrics,
        device=device,
        verbose=True,
        train_assembly=cfg['MODEL']['USE_ASSEMBLY_CLASSIFIER']
    )
    
    # --- Training Loop ---
    max_score = 0
    start_epoch = 0

    if resume:
        checkpoint_path = os.path.join(output_dir, 'latest.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            max_score = checkpoint.get('max_score', 0)
            print(f"Resuming training from epoch {start_epoch}")

    for i in range(start_epoch, cfg['DATASET']['EPOCHS']):
        print(f'\nEpoch: {i}')
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        if i == int(0.75 * cfg['DATASET']['EPOCHS']):
            print("\n[INFO] Manually dropping learning rate by a factor of 10...")
            optimizer.param_groups[0]['lr'] *= 0.1
            print(f"[INFO] New learning rate: {optimizer.param_groups[0]['lr']}\n")

        # Save the latest model
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'max_score': max_score,
        }, os.path.join(output_dir, 'latest.pth'))

        # Save the best model based on validation F-score
        if max_score < valid_logs['fscore']:
            max_score = valid_logs['fscore']
            torch.save(model.state_dict(), os.path.join(output_dir, 'best.pth'))
            print('Best model saved!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Assembly Depth Model')
    parser.add_argument('config', type=str, help='Path to the configuration file.')
    parser.add_argument('--name', type=str, required=True, help='Name for the experiment run.')
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint.')
    args = parser.parse_args()
    main(args.config, args.name, args.resume)
