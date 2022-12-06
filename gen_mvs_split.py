import os
import re
import argparse
import numpy as np
import pandas as pd
import glob
from PIL import Image
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

def gen_split(input_dir, output_dir, split='train'):
    all_scan_ids = os.listdir(input_dir)

    df = pd.read_csv('split.csv')
    scan_ids = df[df['split'] == split]['scanId'].unique()
    split_set = []
    for scan_id in tqdm(scan_ids):
        if scan_id not in all_scan_ids:
            continue
        color_dir = os.path.join(input_dir, scan_id, 'color')
        rgb_files = sorted_alphanum(glob.glob(f'{color_dir}/*.png'))

        num_rgb = len(rgb_files)
        for i in range(0, num_rgb, 1):
            rgb_file = rgb_files[i]
            frame_dict = {}
            rgb_filename = os.path.basename(rgb_file)
            target = int(rgb_filename.split('.')[0].split('-')[-1])
            if i > 0 and i+1 < len(rgb_files):
                target_prev = int(os.path.basename(rgb_files[i-1]).split('.')[0])
                target_next = int(os.path.basename(rgb_files[i+1]).split('.')[0])
            elif i+1 == len(rgb_files):
                target_prev = int(os.path.basename(rgb_files[i-1]).split('.')[0])
                target_next = int(os.path.basename(rgb_files[i-2]).split('.')[0])
            else:
                target_prev = int(os.path.basename(rgb_files[i+1]).split('.')[0])
                target_next = int(os.path.basename(rgb_files[i+2]).split('.')[0])
            refs = [target_prev, target_next]
            frame_dict['scene'] = scan_id
            frame_dict['target'] = target
            frame_dict['refs'] = refs
            split_set.append(frame_dict)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/multiscan_{split}.npy', 'wb') as f:
        np.save(f, split_set, allow_pickle=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mesh reconstruction!')
    parser.add_argument('-i', '--input-dir', type=str, action='store', required=False, help='Input directory')
    parser.add_argument('-o', '--output-dir', type=str, action='store', required=False, help='Output directory')
    args = parser.parse_args()

    gen_split(args.input_dir, args.output_dir, split='train')
    gen_split(args.input_dir, args.output_dir, split='val')
    gen_split(args.input_dir, args.output_dir, split='test')