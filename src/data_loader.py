# src/data_loader.py (Corrected to provide labels for training)

import os
import glob
import yaml
import gzip
import numpy as np
import cv2
import torch
import open3d as o3d

def preprocess(image, intrinsics, target_height, target_width):
    h_orig, w_orig = image.shape[:2]
    scale = min(target_width / w_orig, target_height / h_orig)
    h_new, w_new = int(h_orig * scale), int(w_orig * scale)
    resized_image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_AREA)
    padded_image = np.zeros((target_height, target_width, *image.shape[2:]), dtype=image.dtype)
    pad_top, pad_left = (target_height - h_new) // 2, (target_width - w_new) // 2
    padded_image[pad_top:pad_top + h_new, pad_left:pad_left + w_new] = resized_image
    modified_intrinsics = intrinsics.copy()
    modified_intrinsics[:2, :] *= scale
    modified_intrinsics[0, 2] += pad_left
    modified_intrinsics[1, 2] += pad_top
    return padded_image, modified_intrinsics

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

class NvisiiScene:
    def __init__(self, scene_root_path, labeled=True):
        self.root = scene_root_path
        self.labeled = labeled
        
        all_img_paths = self._find_paths('rgb/*.jpg')
        common_keys = set(all_img_paths.keys())

        if self.labeled:
            all_segmentation_paths = self._find_paths('segmentation/*.npy.gz')
            all_equipment_point_paths = self._find_paths('equipment_points/*.png')
            
            # For labeled data, ensure all corresponding label files exist
            common_keys &= set(all_segmentation_paths.keys())
            common_keys &= set(all_equipment_point_paths.keys())

            self.segmentation_paths = {k: v for k, v in all_segmentation_paths.items() if k in common_keys}
            self.equipment_point_paths = {k: v for k, v in all_equipment_point_paths.items() if k in common_keys}
        
        self.img_paths = {k: v for k, v in all_img_paths.items() if k in common_keys}
        self.equipment_ids = self._parse_equipment_ids()
        self.camera_intrinsics = self._parse_camera_intrinsics()
        self.equipment_pose = self._parse_equipment_pose()
        self.camera_poses = self._parse_camera_poses()

    def _find_paths(self, pattern):
        paths = glob.glob(os.path.join(self.root, pattern))
        return {int(os.path.basename(p).split('.')[0]): p for p in paths}

    def _parse_equipment_ids(self):
        try:
            with open(os.path.join(self.root, 'entity_ids.yaml')) as f: return yaml.safe_load(f)
        except FileNotFoundError: return {}

    def _parse_camera_intrinsics(self):
        intrinsics = {}
        for path in glob.glob(os.path.join(self.root, 'rgb_intrinsics/*.npy')):
            intrinsics[int(os.path.basename(path).split('.')[0])] = np.load(path)
        return intrinsics
        
    def _parse_camera_poses(self):
        poses = {}
        cam_transform = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        for path in glob.glob(os.path.join(self.root, 'rgb_pose/*.npy')):
            pose = np.load(path)
            poses[int(os.path.basename(path).split('.')[0])] = np.dot(pose, cam_transform)
        return poses

    def _parse_equipment_pose(self):
        pose_path = os.path.join(self.root, 'equipment_pose/root/00000.npy')
        return np.load(pose_path) if os.path.exists(pose_path) else np.eye(4)
    
    def get_img_ids(self): return sorted(list(self.img_paths.keys()))
    def get_camera_intrinsics(self, idx): return self.camera_intrinsics.get(idx)
    def get_camera_pose(self, idx): return self.camera_poses.get(idx)
    def get_present_equipment_names(self): return list(self.equipment_ids.keys())
    def get_monodepth(self, idx): return cv2.imread(self.img_paths[idx].replace('rgb','monocular_depth').replace('jpg','png'))
    def get_image(self, idx): return cv2.imread(self.img_paths[idx])
    
    def get_segmentation(self, idx):
        path = self.segmentation_paths.get(idx)
        if path:
            with gzip.GzipFile(path, "r") as f: return np.load(f)
        return None

    def get_equipment_points_norm(self, idx):
        # This logic assumes the equipment points are PNGs in a parallel directory
        path = self.img_paths.get(idx).replace("rgb", "equipment_points").replace(".jpg", ".png")
        img = cv2.imread(path)
        return img / 255.0 if img is not None else None
        
    def get_segmentation_binary(self, idx, ids):
        seg = self.get_segmentation(idx)
        return np.isin(seg, ids).astype(np.uint8) if seg is not None else None

class DatasetPhaseNvisii:
    # This class remains unchanged from the user's provided file
    def __init__(self, path, phase):
        self.root = path
        self.phase = phase
        self.scene_dirs = sorted(glob.glob(os.path.join(self.root, phase, '*')))
        if not self.scene_dirs: raise FileNotFoundError(f"No scenes in: {os.path.join(self.root, phase)}")
        self.mesh_paths = sorted(glob.glob(os.path.join(self.root, 'models/*.obj')))
        self.mesh_names = [os.path.basename(p).split('.obj')[0] for p in self.mesh_paths]
        self._sample_mesh_points()

    def _sample_mesh_points(self, num_points=10000):
        all_meshes = [o3d.io.read_triangle_mesh(p) for p in self.mesh_paths]
        combined_mesh = o3d.geometry.TriangleMesh()
        for mesh in all_meshes:
            if mesh.has_vertices():
                combined_mesh += mesh
        self.sampled_points = combined_mesh.sample_points_uniformly(number_of_points=num_points).points

    def get_overall_sampled_points(self): return np.asarray(self.sampled_points)
    def num_scenes(self): return len(self.scene_dirs)
    def get_scene(self, idx, labeled=True): return NvisiiScene(self.scene_dirs[idx], labeled=labeled)
    def get_mesh_names(self): return self.mesh_names

class ModelPointDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, phase, width=384, height=224, use_monocular_depth=False, epoch_length=200000):
        self.datapath, self.phase = datapath, phase
        self.data_interface = DatasetPhaseNvisii(self.datapath, self.phase)
        self.use_monocular_depth = use_monocular_depth
        self.has_pixel_labels = (phase != 'test')
        self.target_shape = (height, width)

        if self.phase != 'train':
            self.scene_img_pairs = []
            for s_idx in range(self.data_interface.num_scenes()):
                scene = self.data_interface.get_scene(s_idx, labeled=self.has_pixel_labels)
                for img_id in scene.get_img_ids():
                    self.scene_img_pairs.append((s_idx, img_id))
            self.epoch_length = len(self.scene_img_pairs)
        else:
            self.epoch_length = epoch_length

    def __len__(self): return self.epoch_length

    def __getitem__(self, idx):
        try:
            if self.phase == 'train':
                scene_idx = np.random.randint(0, self.data_interface.num_scenes())
                scene = self.data_interface.get_scene(scene_idx, labeled=True)
                if not scene.get_img_ids(): return self.__getitem__(0)
                img_idx = np.random.choice(scene.get_img_ids())
            else:
                if idx >= len(self.scene_img_pairs): return None
                scene_idx, img_idx = self.scene_img_pairs[idx]
                scene = self.data_interface.get_scene(scene_idx, labeled=self.has_pixel_labels)

            if self.use_monocular_depth:
                image = scene.get_monodepth(img_idx)
                if image.shape[2] == 3: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image = scene.get_image(img_idx)
            intrinsics = scene.get_camera_intrinsics(img_idx)
            if image is None or intrinsics is None: return None

            image_processed, intrinsics_modified = preprocess(image, intrinsics, *self.target_shape)

            if self.use_monocular_depth:
                image_processed = np.expand_dims(image_processed, 2)
            image_tensor = to_tensor(image_processed) / 255.0

            equip_names = scene.get_present_equipment_names()
            all_parts = self.data_interface.get_mesh_names()
            assembly_state_truth = np.array([(p in equip_names) for p in all_parts], dtype=np.float32)
            
            # This is the critical block that loads labels for train/val
            if self.has_pixel_labels:
                equip_ids = [scene.equipment_ids[n] for n in equip_names]
                mask = scene.get_segmentation_binary(img_idx, equip_ids)
                pts = scene.get_equipment_points_norm(img_idx)
                if mask is None or pts is None: return None # Skip if labels are missing
                pts[mask == 0] = 0
                
                mask_processed, _ = preprocess(mask, intrinsics, *self.target_shape)
                pts_processed, _ = preprocess(pts, intrinsics, *self.target_shape)
                
                mask_tensor = to_tensor(np.expand_dims(mask_processed, 2))
                model_points_tensor = to_tensor(pts_processed)
            else:
                # Create placeholders for the test phase
                h, w = self.target_shape
                mask_tensor = torch.zeros((1, h, w), dtype=torch.float32)
                model_points_tensor = torch.zeros((3, h, w), dtype=torch.float32)

            return {
                "image": image_tensor,
                "mask": mask_tensor, # Key is now always present
                "model_points_img": model_points_tensor, # Key is now always present
                "assembly_state": assembly_state_truth,
                "intrinsics": intrinsics_modified,
                "gt_equipment_pose_world": scene.equipment_pose,
                "gt_camera_pose_world": scene.get_camera_pose(img_idx),
            }
        except Exception:
            return None
