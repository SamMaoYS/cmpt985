import os
import numpy as np

import hydra
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
import open3d as o3d
from sklearn.neighbors import KDTree
import trimesh

def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances

def evaluate(gt_mesh, pred_mesh):
    # def evaluate(mesh_pred, mesh_trgt, threshold=.05, down_sample=.02):
    mesh_trgt = trimesh.load(gt_mesh)
    mesh_pred = trimesh.load(pred_mesh)

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
        'Chamfer': np.mean(dist2)/2.0 + np.mean(dist1)/2.0,
        'Prec': precision,
        'Recal': recal,
        'F-score': fscore,
    }
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mesh reconstruction!')
    parser.add_argument('-g', '--gt', type=str, action='store', required=False, help='Input gt mesh')
    parser.add_argument('-p', '--pred', type=str, action='store', required=False, help='Input pred mesh')
    args = parser.parse_args()

    evaluate(args.gt, args.pred)
