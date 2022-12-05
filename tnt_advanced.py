import os
import shutil
import glob
import numpy as np
import pdb
import cv2
from PIL import Image

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def write_matrix(filename, matrix):
    with open(filename, 'w+') as f:
        for line in matrix:
            np.savetxt(f, line.reshape(1, -1), fmt='%.6f')

if __name__ == '__main__':
    data_dir = 'tnt_advanced'
    output_dir = 'tnt_data'
    scenes = os.listdir(data_dir)

    for scene in scenes:
        scene_input_dir = os.path.join(data_dir, scene)
        scene_output_dir = os.path.join(output_dir, scene)
        os.makedirs(scene_output_dir, exist_ok=True)

        depth_files = glob.glob(f'{scene_input_dir}/*_depth.npy')
        os.makedirs(os.path.join(scene_output_dir, 'depth'), exist_ok=True)
        for i, depth_file in enumerate(depth_files):
            depth_data = np.load(depth_file)
            depth_data *= 100000
            depth_img = Image.fromarray(depth_data.astype(np.uint16))
            depth_img.save(os.path.join(scene_output_dir, 'depth', f'{i}.png'))

        rgb_files = glob.glob(f'{scene_input_dir}/*_rgb.png')
        os.makedirs(os.path.join(scene_output_dir, 'color'), exist_ok=True)
        for i, rgb_file in enumerate(rgb_files):
            shutil.copy(rgb_file, os.path.join(scene_output_dir, 'color', f'{i}.png'))

        assert len(rgb_files) == len(depth_files)

        camera_dict = np.load(f'{scene_input_dir}/cameras.npz')
        n_images = len(rgb_files)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        idx = 0
        os.makedirs(os.path.join(scene_output_dir, 'pose'), exist_ok=True)
        os.makedirs(os.path.join(scene_output_dir, 'intrinsic'), exist_ok=True)
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)

            if idx == 0:
                extrinsics = np.eye(4)
                rgb_data = np.asarray(Image.open(rgb_file))
                color_height, color_width = rgb_data.shape[0], rgb_data.shape[1]
                depth_height, depth_width = depth_data.shape[0], depth_data.shape[1]

                scale_w = float(depth_width) / color_width
                scale_h = float(depth_height) / color_height
                scale = np.array([scale_w, scale_h, 1.0, 1.0])
                K_depth = np.matmul(np.diag(scale), intrinsics)

                write_matrix(os.path.join(scene_output_dir, 'intrinsic', f'intrinsic_color.txt'), intrinsics)
                write_matrix(os.path.join(scene_output_dir, 'intrinsic', f'intrinsic_depth.txt'), K_depth)
                write_matrix(os.path.join(scene_output_dir, 'intrinsic', 'extrinsic_color.txt'), extrinsics)
                write_matrix(os.path.join(scene_output_dir, 'intrinsic', 'extrinsic_depth.txt'), extrinsics)

            write_matrix(os.path.join(scene_output_dir, 'pose', f'{idx}.txt'), pose)
            idx += 1
            