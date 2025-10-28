#!/usr/bin/env python3
"""
Convert GELLO pickle data to LeRobot dataset format.

Follows the official LeRobot format convention:
- Images: HWC format (Height, Width, Channel)
- Keys: 'state', 'actions', 'image', 'wrist_image'
- Explicit feature definitions with dtype, shape, names

Usage:
    python experiments/convert_gello_lerobot.py \
        --input-dir data/GelloAgent/1028_143052 \
        --output-dir lerobot_datasets/my_robot_data \
        --repo-id "username/my_robot_dataset" \
        --fps 30

Or process multiple episodes:
    python experiments/convert_gello_lerobot.py \
        --input-dir data/GelloAgent \
        --output-dir lerobot_datasets/all_episodes \
        --repo-id "username/my_robot_dataset" \
        --fps 30 \
        --recursive

Notes:
- Supports both old naming (wrist_rgb/side_rgb) and new naming (wrist_image/image)
- Output keys: 'image' (main camera), 'wrist_image' (wrist camera)
- Images stored in HWC format for LeRobot compatibility
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_gello_frame(pkl_path: Path) -> Dict:
    """Load a single GELLO pickle frame."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def convert_gello_episode_to_lerobot(
    pkl_files: List[Path],
    episode_index: int = 0,
    fps: int = 30,
) -> Dict[str, torch.Tensor]:
    """
    Convert a single GELLO episode (list of pickle files) to LeRobot format.
    
    GELLO format (per frame):
    {
        'joint_positions': np.ndarray (7,),
        'joint_velocities': np.ndarray (7,),
        'ee_pos_quat': np.ndarray (7,),
        'gripper_position': np.ndarray (1,),
        'wrist_rgb': np.ndarray (H, W, 3),  or 'image'
        'side_rgb': np.ndarray (H, W, 3),   or 'wrist_image'
        'control': np.ndarray (7,)  # action
    }
    
    LeRobot format (per episode):
    {
        'state': torch.Tensor (T, state_dim),
        'actions': torch.Tensor (T, action_dim),
        'image': torch.Tensor (T, H, W, 3),          # main camera (side view)
        'wrist_image': torch.Tensor (T, H, W, 3),    # wrist camera
        'episode_index': torch.Tensor (T,),
        'frame_index': torch.Tensor (T,),
        'timestamp': torch.Tensor (T,),
        'next.done': torch.Tensor (T,),
    }
    """
    print(f"Processing {len(pkl_files)} frames for episode {episode_index}...")
    
    # Lists to accumulate data
    states = []
    actions = []
    wrist_images = []
    side_images = []
    timestamps = []
    
    # Process each frame
    for i, pkl_file in enumerate(tqdm(pkl_files, desc=f"Episode {episode_index}")):
        try:
            frame = load_gello_frame(pkl_file)
            
            # State: joint positions (can extend with velocities, ee_pos, etc.)
            state = frame['joint_positions'].astype(np.float32)
            states.append(state)
            
            # Action: control command
            action = frame['control'].astype(np.float32)
            actions.append(action)
            
            # Camera images - keep in HWC format (LeRobot standard)
            # Check both old naming (wrist_rgb/side_rgb) and new naming (image/wrist_image)
            if 'wrist_image' in frame:
                wrist_rgb = frame['wrist_image']  # (H, W, 3)
            elif 'wrist_rgb' in frame:
                wrist_rgb = frame['wrist_rgb']  # (H, W, 3)
            else:
                wrist_rgb = None
            
            if wrist_rgb is not None:
                wrist_images.append(wrist_rgb.astype(np.uint8))
            
            if 'image' in frame:
                side_rgb = frame['image']  # (H, W, 3)
            elif 'side_rgb' in frame:
                side_rgb = frame['side_rgb']  # (H, W, 3)
            else:
                side_rgb = None
            
            if side_rgb is not None:
                side_images.append(side_rgb.astype(np.uint8))
            
            # Timestamp (frame number / fps)
            timestamps.append(i / fps)
            
        except Exception as e:
            print(f"Warning: Failed to process {pkl_file}: {e}")
            continue
    
    if len(states) == 0:
        raise ValueError(f"No valid frames found in episode {episode_index}")
    
    # Convert to torch tensors
    num_frames = len(states)
    
    lerobot_data = {
        'state': torch.from_numpy(np.stack(states)),
        'actions': torch.from_numpy(np.stack(actions)),
        'episode_index': torch.full((num_frames,), episode_index, dtype=torch.int64),
        'frame_index': torch.arange(num_frames, dtype=torch.int64),
        'timestamp': torch.tensor(timestamps, dtype=torch.float32),
        'next.done': torch.zeros(num_frames, dtype=torch.bool),
    }
    
    # Mark last frame as done
    lerobot_data['next.done'][-1] = True
    
    # Add camera images if available (HWC format)
    if wrist_images:
        lerobot_data['wrist_image'] = torch.from_numpy(
            np.stack(wrist_images)  # (T, H, W, 3)
        )
    
    if side_images:
        lerobot_data['image'] = torch.from_numpy(
            np.stack(side_images)  # (T, H, W, 3)
        )
    
    print(f"Episode {episode_index}: {num_frames} frames")
    print(f"  State shape: {lerobot_data['state'].shape}")
    print(f"  Actions shape: {lerobot_data['actions'].shape}")
    if wrist_images:
        print(f"  Wrist image shape: {lerobot_data['wrist_image'].shape}")
    if side_images:
        print(f"  Image shape: {lerobot_data['image'].shape}")
    
    return lerobot_data


def find_episodes(input_dir: Path, recursive: bool = False) -> List[List[Path]]:
    """
    Find all episodes in the input directory.
    
    Returns:
        List of episodes, where each episode is a list of pickle file paths.
    """
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    episodes = []
    
    if recursive:
        # Look for subdirectories, each is an episode
        subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
        print(f"Found {len(subdirs)} episode directories")
        
        for subdir in sorted(subdirs):
            pkl_files = sorted(subdir.glob("*.pkl"))
            if len(pkl_files) > 0:
                episodes.append(pkl_files)
                print(f"  {subdir.name}: {len(pkl_files)} frames")
    else:
        # Single episode directory
        pkl_files = sorted(input_dir.glob("*.pkl"))
        if len(pkl_files) > 0:
            episodes.append(pkl_files)
            print(f"Found {len(pkl_files)} frames in {input_dir}")
    
    return episodes


def save_lerobot_dataset(
    episodes_data: List[Dict[str, torch.Tensor]],
    output_dir: Path,
    repo_id: str,
    fps: int,
):
    """Save episodes in LeRobot dataset format."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        use_lerobot = True
    except ImportError:
        print("Warning: lerobot not installed. Saving as raw PyTorch tensors.")
        use_lerobot = False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if use_lerobot:
        # Use official LeRobot dataset format
        print(f"\nCreating LeRobot dataset at {output_dir}...")
        
        # Infer shapes from first episode
        first_ep = episodes_data[0]
        state_dim = first_ep['state'].shape[1]
        action_dim = first_ep['actions'].shape[1]
        
        # Build features dict with proper dtype and shape
        features = {
            "state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
        }
        
        # Add image features if available
        if 'image' in first_ep:
            img_data = first_ep['image']
            H, W, C = img_data.shape[1], img_data.shape[2], img_data.shape[3]
            features["image"] = {
                "dtype": "image",
                "shape": (H, W, C),
                "names": ["height", "width", "channel"],
            }
            print(f"  Main camera 'image': {H}x{W}x{C}")
        
        if 'wrist_image' in first_ep:
            wrist_data = first_ep['wrist_image']
            H, W, C = wrist_data.shape[1], wrist_data.shape[2], wrist_data.shape[3]
            features["wrist_image"] = {
                "dtype": "image",
                "shape": (H, W, C),
                "names": ["height", "width", "channel"],
            }
            print(f"  Wrist camera 'wrist_image': {H}x{W}x{C}")
        
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        
        # Create dataset with explicit features
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type="ur",
            fps=fps,
            root=output_dir,
            features=features,
            image_writer_threads=10,
            image_writer_processes=5,
        )
        
        # Add episodes
        for ep_idx, ep_data in enumerate(episodes_data):
            dataset.add_episode(ep_data, episode_index=ep_idx)
            print(f"  Added episode {ep_idx} ({len(ep_data['actions'])} frames)")
        
        dataset.consolidate()
        print(f"✓ LeRobot dataset saved to {output_dir}")
        
    else:
        # Save as PyTorch tensors (fallback)
        print(f"\nSaving episodes as PyTorch tensors to {output_dir}...")
        
        for ep_idx, ep_data in enumerate(episodes_data):
            ep_path = output_dir / f"episode_{ep_idx:04d}.pt"
            torch.save(ep_data, ep_path)
            print(f"  Saved episode {ep_idx} to {ep_path}")
        
        # Save metadata
        metadata = {
            'num_episodes': len(episodes_data),
            'fps': fps,
            'repo_id': repo_id,
        }
        torch.save(metadata, output_dir / "metadata.pt")
        print(f"✓ Dataset saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert GELLO pickle data to LeRobot format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing GELLO pickle files or subdirectories of episodes",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for LeRobot dataset",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="gello/ur_robot_data",
        help="Repository ID for the dataset (e.g., 'username/dataset_name')",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second of the recorded data",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process all subdirectories as separate episodes",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=10,
        help="Minimum number of frames required for an episode",
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("GELLO to LeRobot Converter")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Repository ID: {args.repo_id}")
    print(f"FPS: {args.fps}")
    print(f"Recursive: {args.recursive}")
    print("=" * 60)
    
    # Find all episodes
    episodes = find_episodes(input_dir, recursive=args.recursive)
    
    if len(episodes) == 0:
        print("Error: No episodes found!")
        return
    
    # Filter by minimum frames
    episodes = [ep for ep in episodes if len(ep) >= args.min_frames]
    print(f"\nProcessing {len(episodes)} episodes (min {args.min_frames} frames)...")
    
    # Convert each episode
    episodes_data = []
    for ep_idx, pkl_files in enumerate(episodes):
        try:
            ep_data = convert_gello_episode_to_lerobot(
                pkl_files,
                episode_index=ep_idx,
                fps=args.fps,
            )
            episodes_data.append(ep_data)
        except Exception as e:
            print(f"Error processing episode {ep_idx}: {e}")
            continue
    
    if len(episodes_data) == 0:
        print("Error: No episodes successfully converted!")
        return
    
    # Save dataset
    save_lerobot_dataset(
        episodes_data,
        output_dir,
        args.repo_id,
        args.fps,
    )
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Total episodes: {len(episodes_data)}")
    total_frames = sum(len(ep['state']) for ep in episodes_data)
    print(f"Total frames: {total_frames}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

