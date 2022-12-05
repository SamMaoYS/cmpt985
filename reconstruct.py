import os
import re
import argparse
import numpy as np
import glob
from PIL import Image
import open3d as o3d
from tqdm import tqdm
import pdb

def sorted_alphanum(file_list):
    """sort the file list by arrange the numbers in filenames in increasing order
    :param file_list: a file list
    :return: sorted file list
    """
    if len(file_list) <= 1:
        return file_list, [0]

    def convert(text): return int(text) if text.isdigit() else text
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(file_list, key=alphanum_key)

class Reconstruct:
    def __init__(self):
        self.rgb_files = []
        self.depth_files = []
        self.pose_files = []
        self.intrinsics = np.eye(4)

    def read_intrinsics(self, filename):
        self.intrinsics = np.loadtxt(filename)[:3, :3]

    def rgb_input(self, input_dir):
        self.rgb_files = sorted_alphanum(glob.glob(f'{input_dir}/*.png'))

    def depth_input(self, input_dir):
        self.depth_files = sorted_alphanum(glob.glob(f'{input_dir}/*.png'))

    def pose_input(self, input_dir):
        self.pose_files = sorted_alphanum(glob.glob(f'{input_dir}/*.txt'))

    @staticmethod
    def align_color2depth(o3d_color, o3d_depth, fast=False):
        # use metadata to get depth map size can be faster
        color_data = np.asarray(o3d_color)
        depth_data = np.asarray(o3d_depth)
        if np.shape(color_data)[0:2] != np.shape(depth_data)[0:2]:
            color = Image.fromarray(color_data)
            depth = Image.fromarray(depth_data)
            if fast:
                color = color.resize(depth.size, Image.NEAREST)
            else:
                color = color.resize(depth.size)
            return o3d.geometry.Image(np.asarray(color))
        return o3d_color

    def integrate(self, output, voxel_length=0.01, sdf_trunc=0.08):
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            volume_unit_resolution=8
        )

        with open('scene_00000_00/scene_00000_00.align.json') as f:
            import json
            align_data = json.load(f)
            align_transform = np.asarray(align_data['coordinate_transform']).reshape((4,4), order='F')

        for i, pose_file in enumerate(tqdm(self.pose_files)):
            color = o3d.io.read_image(self.rgb_files[i])
            depth = o3d.io.read_image(self.depth_files[i])
            color = self.align_color2depth(color, depth)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        color,
                        depth,
                        depth_scale=1000.0,
                        depth_trunc=5.0,
                        convert_rgb_to_intensity=False)
            np_depth = np.asarray(depth)
            K = self.intrinsics.flatten('F').tolist()
            o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=np_depth.shape[1], height=np_depth.shape[0], fx=K[0], fy=K[4], cx=K[6], cy=K[7])
            pose = np.loadtxt(pose_file)
            extrinsics = np.linalg.inv(pose)
            volume.integrate(rgbd, o3d_intrinsic, extrinsics)

        mesh = volume.extract_triangle_mesh()
        if os.path.dirname(output):
            os.makedirs(os.path.dirname(output), exist_ok=True)
        success = o3d.io.write_triangle_mesh(output, mesh, write_vertex_normals=True)
        return success

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mesh reconstruction!')
    parser.add_argument('-c', '--color', type=str, action='store', required=False, help='Input color directory')
    parser.add_argument('-d', '--depth', type=str, action='store', required=False, help='Input depth directory')
    parser.add_argument('-p', '--pose', type=str, action='store', required=False, help='Input pose directory')
    parser.add_argument('-i', '--intrinsics', type=str, action='store', required=False, help='Input intrinsics directory')
    parser.add_argument('--voxel-length', type=float, default=0.01, action='store', required=False, help='Parameter voxel_length')
    parser.add_argument('--sdf-trunc', type=float, default=0.08, action='store', required=False, help='Parameter sdf_trunc')
    parser.add_argument('-o', '--output-file', type=str, action='store', required=False, help='Output mesh file path')
    args = parser.parse_args()

    recon = Reconstruct()
    recon.read_intrinsics(args.intrinsics)
    recon.rgb_input(args.color)
    recon.depth_input(args.depth)
    recon.pose_input(args.pose)
    success = recon.integrate(args.output_file, args.voxel_length, args.sdf_trunc)
    if success:
        print(f'Reconstructed mesh saved into {args.output_file}')
    else:
        print(f'Reconstruction failed')
