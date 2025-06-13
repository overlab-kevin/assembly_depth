# test.py (Revised for Single-Image Pose & Assembly Evaluation)

import os
import yaml
import argparse
import torch
import numpy as np
np.set_printoptions(precision=4, suppress=True)
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from src.data_loader import ModelPointDataset
from src.model import UnetReg
from src.utils import (
    get_assembly_aux_params,
    get_3d_correspondences,
    estimate_pose_from_correspondences,
    calculate_msd_error
)

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

def main(experiment_dir, test_data_path):
    # --- Setup ---
    config_path = os.path.join(experiment_dir, 'config.yaml')
    weights_path = os.path.join(experiment_dir, 'best.pth')
    with open(config_path) as f: cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- DataLoader ---
    test_dataset = ModelPointDataset(test_data_path, 'test', cfg['MODEL']['IMG_SIZE'][1], cfg['MODEL']['IMG_SIZE'][0], cfg['MODEL']['USE_MONOCULAR_DEPTH'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['DATASET']['BATCH_SIZE'], num_workers=cfg['DATASET']['WORKERS'], collate_fn=collate_fn)
    print(f"Loaded {len(test_dataset)} samples from the test set.")
    
    # --- Model ---
    num_parts = len(test_dataset.data_interface.get_mesh_names())
    aux_params = get_assembly_aux_params(num_parts) if cfg['MODEL']['USE_ASSEMBLY_CLASSIFIER'] else None
    in_channels = 1 if cfg['MODEL']['USE_MONOCULAR_DEPTH'] else 3
    model = UnetReg(cfg['MODEL']['ENCODER_NAME'], 3, 1, aux_params=aux_params).to(device)
    
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model weights from {weights_path}")

    # --- Evaluation Loop ---
    all_msd_errors = []
    all_assembly_accuracies = []
    mesh_points = test_dataset.data_interface.get_overall_sampled_points()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Set"):
            if batch is None: continue
            
            images = batch['image'].to(device)
            # expand channels from 1 to 3 if using monocular depth
            if cfg['MODEL']['USE_MONOCULAR_DEPTH'] and images.size(1) == 1:
                images = images.expand(-1, 3, -1, -1)
            pred_mask_all, pred_positions, classifications = model(images)
            
            # Un-normalize predicted points from [0, 1] range to original model coordinates
            mesh_min_bound = mesh_points.min(axis=0)
            mesh_max_bound = mesh_points.max(axis=0)
            
            # Process each item in the batch
            for i in range(images.size(0)):
                # --- Get predictions ---
                pred_mask = pred_mask_all[i].squeeze().cpu().numpy().round()
                pred_points_normalized = pred_positions[i, 0:3].permute(1, 2, 0).cpu().numpy()
                pred_depth = pred_positions[i, 4].cpu().numpy()
                
                # Un-normalize model points
                pred_points_unnormalized = pred_points_normalized * (mesh_max_bound - mesh_min_bound) + mesh_min_bound

                # --- Get ground truth for this image ---
                intrinsics = batch['intrinsics'][i].numpy()
                gt_equip_pose_world = batch['gt_equipment_pose_world'][i].numpy()
                gt_cam_pose_world = batch['gt_camera_pose_world'][i].numpy()
                
                # --- Estimate Predicted Pose ---
                obj_pts, cam_pts = get_3d_correspondences(pred_mask, pred_points_unnormalized, pred_depth, intrinsics)
                pred_pose_cam = estimate_pose_from_correspondences(obj_pts, cam_pts)

                # --- Calculate Ground Truth Pose in Camera Frame ---
                # T_cam_world = inv(T_world_cam)
                # T_equip_cam = T_cam_world * T_equip_world
                gt_pose_cam = np.linalg.inv(gt_cam_pose_world) @ gt_equip_pose_world
                
                # --- Calculate Errors ---
                msd_error = calculate_msd_error(gt_pose_cam, pred_pose_cam, mesh_points)
                all_msd_errors.append(msd_error)

                if classifications is not None:
                    gt_assembly_state = batch['assembly_state'][i].numpy()
                    pred_assembly_state = (torch.sigmoid(classifications[i]).cpu().numpy() > 0.5).astype(int)
                    acc = accuracy_score(gt_assembly_state, pred_assembly_state)
                    all_assembly_accuracies.append(acc)

    # --- Report Final Metrics ---
    print("\n--- Final Evaluation Results ---")
    print(f"Results computed over {len(all_msd_errors)} images.")
    
    print("\nPose Estimation (MSD Error in meters):")
    print(f"  Median (50th percentile): {np.percentile(all_msd_errors, 50):.4f}")
    print(f"  75th percentile:          {np.percentile(all_msd_errors, 75):.4f}")
    print(f"  90th percentile:          {np.percentile(all_msd_errors, 90):.4f}")
    
    if all_assembly_accuracies:
        print("\nAssembly State Recognition:")
        print(f"  Mean Part Accuracy:       {np.mean(all_assembly_accuracies):.4f}")
    print("--------------------------------\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Assembly Depth Model')
    parser.add_argument('--experiment_dir', required=True, help='Path to experiment dir with config and weights.')
    parser.add_argument('--test_data_path', required=True, help='Path to the root of the test dataset.')
    args = parser.parse_args()
    main(args.experiment_dir, args.test_data_path)
