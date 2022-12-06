import os

import numpy as np

import hydra
from omegaconf import DictConfig
from PIL import Image
import open3d as o3d
from sklearn.neighbors import KDTree
import trimesh
import pdb

class Reconstruct:
    def __init__(self, cfg):
        self.cfg = cfg

    def integrate(self):
        cfg = self.cfg

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.01,
            sdf_trunc=0.05,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            volume_unit_resolution=8
        )

        depth_npy_dir = cfg.depth_npy_dir
        depth_npy_files = os.listdir(depth_npy_dir)
        for depth_npy_file in depth_npy_files:
            depth_npy_path = os.path.join(depth_npy_dir, depth_npy_file)
            depth_npy = np.load(depth_npy_path, allow_pickle=True).tolist()
            color = depth_npy['color'].astype(np.uint8)
            depth_pred = depth_npy['depth_pred'].astype(np.float32)
            frame_name = depth_npy['fID']
            rgb = o3d.geometry.Image(color)
            depth = o3d.geometry.Image(depth_pred)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        rgb,
                        depth,
                        depth_scale=1.0,
                        depth_trunc=5.0,
                        convert_rgb_to_intensity=False)

            inv_K = depth_npy['inv_K']
            K = np.linalg.inv(inv_K).flatten('F').tolist()
            o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width = depth_pred.shape[1], height = depth_pred.shape[0], fx = K[0], fy = K[4], cx = K[6], cy = K[7])
            pose = depth_npy['pose']
            volume.integrate(rgbd, o3d_intrinsic, pose)

        mesh = volume.extract_triangle_mesh()
        success = o3d.io.write_triangle_mesh(f'{cfg.scene_id}.ply', mesh, write_vertex_normals=True)

def gen_val_split(cfg):
    scene_id = cfg.scene_id
    rgb_filenames = os.listdir(cfg.rgb_dir)

    val_split = []
    for rgb_filename in rgb_filenames:
        frame_dict = {}
        target = int(rgb_filename.split('.')[0].split('-')[-1])
        refs = [target-1, target+1]
        frame_dict['scene'] = scene_id
        frame_dict['target'] = target
        frame_dict['refs'] = refs
        val_split.append(frame_dict)

    with open(f'{scene_id}.npy', 'wb') as f:
        np.save(f, val_split, allow_pickle=True)

def gen_frames(cfg):
    depth_npy_dir = cfg.depth_npy_dir
    output_dir = cfg.output_dir

    os.makedirs(output_dir, exist_ok=True)

    depth_npy_files = os.listdir(depth_npy_dir)
    for i, depth_npy_file in enumerate(depth_npy_files):
        depth_npy_path = os.path.join(depth_npy_dir, depth_npy_file)
        depth_npy = np.load(depth_npy_path, allow_pickle=True).tolist()
        color = depth_npy['color']
        depth_pred = depth_npy['depth_pred']*1000
        depth_img = Image.fromarray(depth_pred.astype(np.uint16))
        frame_name = depth_npy['fID']
        depth_img.save(os.path.join(output_dir, cfg.depth_frame % i))
        color_img = Image.fromarray(color.astype(np.uint8))
        color_img.save(os.path.join(output_dir, cfg.rgb_frame % i))
        
        pose = depth_npy['pose']
        pose = np.linalg.inv(pose)
        mat = np.matrix(pose)
        with open(os.path.join(output_dir, cfg.pose_frame % i),'w+') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%1.4e')

def gen_intrinsics(cfg):
    depth_npy_dir = cfg.depth_npy_dir
    depth_npy_files = os.listdir(depth_npy_dir)

    for i, depth_npy_file in enumerate(depth_npy_files):
        depth_npy_path = os.path.join(depth_npy_dir, depth_npy_file)
        depth_npy = np.load(depth_npy_path, allow_pickle=True).tolist()
        inv_K = depth_npy['inv_K']
        K = np.linalg.inv(inv_K)

        mat = np.matrix(K)
        with open(cfg.intrinsics, 'w+') as f:
            for line in mat:
                np.savetxt(f, line)
        break

def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances

def evaluate(cfg):
    # def evaluate(mesh_pred, mesh_trgt, threshold=.05, down_sample=.02):
    mesh_trgt = trimesh.load(cfg.gt_mesh)
    mesh_pred = trimesh.load(cfg.pred_mesh)

    threshold = 0.05
    down_sample = 0.02
    pcd_trgt = o3d.geometry.PointCloud()
    pcd_pred = o3d.geometry.PointCloud()
    
    pcd_trgt.points = o3d.utility.Vector3dVector(mesh_trgt.vertices[:, :3])
    pcd_pred.points = o3d.utility.Vector3dVector(mesh_pred.vertices[:, :3])

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    dist1 = nn_correspondance(verts_pred, verts_trgt)
    dist2 = nn_correspondance(verts_trgt, verts_pred)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {
        'Acc': np.mean(dist2),
        'Comp': np.mean(dist1),
        'Prec': precision,
        'Recal': recal,
        'F-score': fscore,
    }
    return metrics

@hydra.main(config_path="config", config_name="config", version_base='1.2')
def main(cfg: DictConfig):
    gen_val_split(cfg)

    # recons = Reconstruct(cfg)
    # recons.integrate()
    # gen_intrinsics(cfg)
    # gen_frames(cfg)

    # metrics = evaluate(cfg)
    # print(metrics)


if __name__ == '__main__':
    main()