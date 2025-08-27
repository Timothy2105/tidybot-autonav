import argparse
import os
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import glob
import torch
import json


def convert_npz_to_image(npz_path, output_dir, use_uimg=True):
    try:
        # load NPZ file
        data = np.load(npz_path)
        
        # extract image data
        if use_uimg and 'uimg' in data:
            # uimg is typically HxWx3 format
            img_data = data['uimg'] # (H, W, 3)
            # ensure it's in the right format and range
            if img_data.dtype == np.float32 or img_data.dtype == np.float64:
                if img_data.max() <= 1.0:
                    img_data = (img_data * 255).astype(np.uint8)
                else:
                    img_data = img_data.astype(np.uint8)
            
        elif 'img' in data:
            img_data = data['img']
            if img_data.shape[0] == 3:
                img_data = np.transpose(img_data, (1, 2, 0))
            
            if img_data.dtype == np.float32 or img_data.dtype == np.float64:
                if img_data.max() <= 1.0:
                    img_data = (img_data * 255).astype(np.uint8)
                else:
                    img_data = img_data.astype(np.uint8)
        else:
            print(f"Warning: No image data found in {npz_path}")
            return False
        
        # get frame info
        frame_id = data.get('frame_id', 0)
        timestamp = data.get('timestamp', 0)
        
        # extract keyframe number
        filename = Path(npz_path).stem
        kf_number = filename.split('_')[-1] if '_' in filename else '000000'
        
        # create output filename
        output_filename = f"keyframe_{kf_number}_frame{frame_id:06d}.png"
        output_path = output_dir / output_filename
        
        # convert BGR to RGB 
        if len(img_data.shape) == 3 and img_data.shape[2] == 3:
            img_data_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        else:
            img_data_bgr = img_data
        
        # save image
        success = cv2.imwrite(str(output_path), img_data_bgr)
        
        if success:
            print(f"Converted {npz_path} -> {output_path}")
            return True
        else:
            print(f"Failed to save {output_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {npz_path}: {e}")
        return False


def extract_keyframe_poses(input_dir):
    """Extract keyframe poses from keyframes.pth and save as JSON."""
    input_dir = Path(input_dir)
    keyframes_file = input_dir / "keyframes.pth"
    
    if not keyframes_file.exists():
        print(f"Warning: keyframes.pth not found in {input_dir}")
        return False
    
    try:
        print(f"Loading keyframe poses from: {keyframes_file}")
        
        # Load the keyframes data
        keyframes_data = torch.load(keyframes_file, map_location='cpu')
        
        print(f"Found {keyframes_data['n_keyframes']} keyframes")
        
        # Extract pose information
        poses_info = []
        
        # Method 1: From trajectory_data (if available)
        if 'trajectory_data' in keyframes_data and 'poses' in keyframes_data['trajectory_data']:
            print("Extracting poses from trajectory_data...")
            poses = keyframes_data['trajectory_data']['poses']
            timestamps = keyframes_data['trajectory_data']['pose_timestamps']
            
            for i, (pose, timestamp) in enumerate(zip(poses, timestamps)):
                # pose is a 7-DOF Sim3: [x, y, z, qx, qy, qz, qw]
                pose_array = np.array(pose).flatten()
                
                pose_info = {
                    'keyframe_index': i,
                    'timestamp': float(timestamp),
                    'position': {
                        'x': float(pose_array[0]),
                        'y': float(pose_array[1]), 
                        'z': float(pose_array[2])
                    },
                    'orientation_quaternion': {
                        'qx': float(pose_array[3]),
                        'qy': float(pose_array[4]),
                        'qz': float(pose_array[5]),
                        'qw': float(pose_array[6])
                    }
                }
                
                # Add scale if it's Sim3 (8 dimensions)
                if len(pose_array) > 7:
                    pose_info['scale'] = float(pose_array[7])
                
                poses_info.append(pose_info)
        
        # Method 2: From individual keyframe_data (if trajectory_data not available)
        elif 'keyframe_data' in keyframes_data:
            print("Extracting poses from individual keyframe_data...")
            
            for kf_data in keyframes_data['keyframe_data']:
                pose_array = np.array(kf_data['T_WC']).flatten()
                
                pose_info = {
                    'keyframe_index': kf_data['keyframe_index'],
                    'frame_id': kf_data['frame_id'],
                    'timestamp': float(kf_data.get('insertion_timestamp', 0)),
                    'position': {
                        'x': float(pose_array[0]),
                        'y': float(pose_array[1]),
                        'z': float(pose_array[2])
                    },
                    'orientation_quaternion': {
                        'qx': float(pose_array[3]),
                        'qy': float(pose_array[4]),
                        'qz': float(pose_array[5]),
                        'qw': float(pose_array[6])
                    }
                }
                
                # Add scale if it's Sim3
                if len(pose_array) > 7:
                    pose_info['scale'] = float(pose_array[7])
                
                poses_info.append(pose_info)
        
        else:
            print("Warning: Could not find pose data in keyframes.pth")
            return False
        
        # Save poses as JSON
        poses_file = input_dir / "kf_poses.json"
        with open(poses_file, 'w') as f:
            json.dump(poses_info, f, indent=2)
        
        print(f"Saved {len(poses_info)} keyframe poses to: {poses_file}")
        
        # Print summary
        if poses_info:
            print(f"  Time span: {poses_info[0]['timestamp']:.2f}s to {poses_info[-1]['timestamp']:.2f}s")
            
            # Calculate trajectory length
            total_distance = 0
            for i in range(1, len(poses_info)):
                prev_pos = poses_info[i-1]['position']
                curr_pos = poses_info[i]['position']
                dist = np.sqrt((curr_pos['x'] - prev_pos['x'])**2 + 
                              (curr_pos['y'] - prev_pos['y'])**2 + 
                              (curr_pos['z'] - prev_pos['z'])**2)
                total_distance += dist
            
            print(f"  Total trajectory length: {total_distance:.2f}m")
        
        return True
        
    except Exception as e:
        print(f"Error extracting keyframe poses: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert keyframe NPZ files to images')
    parser.add_argument('--dir', '-d', required=True, 
                       help='Directory containing keyframe NPZ files')
    parser.add_argument('--use-processed', action='store_true',
                       help='Use processed images (img) instead of original images (uimg)')
    
    args = parser.parse_args()
    
    # validate input directory
    input_dir = Path(args.dir)
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return 1
    
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        return 1
    
    # create output directory
    output_dir = input_dir / "kf-imgs"
    output_dir.mkdir(exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # find all keyframe NPZ files
    npz_pattern = str(input_dir / "keyframe_*.npz")
    npz_files = glob.glob(npz_pattern)
    
    if not npz_files:
        print(f"No keyframe NPZ files found in {input_dir}")
        return 1
    
    # sort files by keyframe number
    npz_files.sort()
    
    print(f"Found {len(npz_files)} keyframe NPZ files")
    
    # convert files
    use_uimg = not args.use_processed
    successful_conversions = 0
    
    for npz_file in tqdm(npz_files, desc="Converting keyframes"):
        if convert_npz_to_image(npz_file, output_dir, use_uimg=use_uimg):
            successful_conversions += 1
    
    print(f"\nConversion complete!")
    print(f"Successfully converted {successful_conversions}/{len(npz_files)} files")
    print(f"Images saved to: {output_dir}")
    
    # Also extract keyframe poses
    print(f"\nExtracting keyframe poses...")
    poses_extracted = extract_keyframe_poses(input_dir)
    
    if poses_extracted:
        print(f"Keyframe poses saved to: {input_dir / 'kf_poses.json'}")
    else:
        print(f"Failed to extract keyframe poses")
    
    return 0


if __name__ == "__main__":
    exit(main()) 