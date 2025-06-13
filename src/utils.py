# src/utils.py

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

def get_assembly_aux_params(num_parts: int) -> dict:
    """Creates the dictionary of auxiliary parameters for the model's classification head."""
    return dict(pooling='avg', dropout=None, activation=None, classes=num_parts)

def get_3d_correspondences(pred_mask, pred_points, pred_depth, intrinsics):
    """
    For a single image, generates 3D points in the object's coordinate system
    and their corresponding 3D points in the camera's coordinate system.
    
    Args:
        pred_mask (np.array): The predicted binary segmentation mask (H x W).
        pred_points (np.array): The predicted normalized model points (H x W x 3).
        pred_depth (np.array): The predicted metric depth in meters (H x W).
        intrinsics (np.array): The 3x3 camera intrinsics matrix for this image.

    Returns:
        tuple: (object_points, camera_points)
    """
    # Find pixel coordinates of the segmented object
    mask_coords = np.column_stack(np.where(pred_mask > 0))
    if mask_coords.shape[0] == 0:
        return None, None

    # Downsample for efficiency if there are too many points
    if mask_coords.shape[0] > 1000:
        indices = np.random.choice(mask_coords.shape[0], 1000, replace=False)
        mask_coords = mask_coords[indices, :]

    # Get the predicted depth and model points at these coordinates
    depth_values = pred_depth[mask_coords[:, 0], mask_coords[:, 1]] * 2.0
    object_points = pred_points[mask_coords[:, 0], mask_coords[:, 1], :]

    # Unproject pixels to 3D camera coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # (u,v) -> (y,x) in numpy array indexing
    u, v = mask_coords[:, 1], mask_coords[:, 0]
    
    camera_points_x = (u - cx) * depth_values / fx
    camera_points_y = (v - cy) * depth_values / fy
    camera_points_z = depth_values
    
    camera_points = np.vstack([camera_points_x, camera_points_y, camera_points_z]).T
    
    return object_points, camera_points

def estimate_pose_from_correspondences(all_object_points, all_camera_points):
    """
    Estimates the 6D pose (object-to-camera) using RANSAC on 3D-3D correspondences.
    """
    if all_object_points is None or all_camera_points is None or len(all_object_points) < 3:
        return np.eye(4)

    # Use Open3D's RANSAC for robust pose estimation
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(all_object_points)
    
    cam_pcd = o3d.geometry.PointCloud()
    cam_pcd.points = o3d.utility.Vector3dVector(all_camera_points)

    corres = o3d.utility.Vector2iVector(np.array([np.arange(len(all_object_points)), np.arange(len(all_object_points))]).T)

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        obj_pcd, cam_pcd, corres,
        max_correspondence_distance=0.1,  # 10cm threshold
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=3,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    
    return result.transformation

def calculate_msd_error(pose_gt, pose_pred, model_surface_points):
    """
    Calculates the Maximum Surface Distance (MSD) error.
    
    Args:
        pose_gt (np.array): The 4x4 ground truth pose matrix.
        pose_pred (np.array): The 4x4 predicted pose matrix.
        model_surface_points (np.array): An array of (N, 3) points sampled from the object's surface.

    Returns:
        float: The MSD error in meters.
    """
    points_h = np.hstack((model_surface_points, np.ones((model_surface_points.shape[0], 1))))
    
    # Transform points by both ground truth and prediction
    points_gt_transformed = (pose_gt @ points_h.T).T[:, :3]
    points_pred_transformed = (pose_pred @ points_h.T).T[:, :3]
    
    # Calculate Euclidean distance between transformed points
    error_distances = np.linalg.norm(points_gt_transformed - points_pred_transformed, axis=1)
    
    return np.max(error_distances)
