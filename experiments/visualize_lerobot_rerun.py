#!/usr/bin/env python3
"""
Visualize converted LeRobot dataset using Rerun.

Visualizes PyTorch tensor format created by convert_gello_lerobot.py

Installation:
    pip install rerun-sdk torch

Usage:
    # Visualize a single episode
    python experiments/visualize_lerobot_rerun.py lerobot_datasets/demo/episode_0000.pt

    # Save recording for later viewing
    python experiments/visualize_lerobot_rerun.py lerobot_datasets/demo/episode_0000.pt --save demo.rrd

    # View all episodes in a dataset
    python experiments/visualize_lerobot_rerun.py lerobot_datasets/demo --all

    # View saved recording
    rerun demo.rrd
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import rerun as rr
import torch


def load_lerobot_episode(episode_path: Path):
    """Load a single LeRobot episode PyTorch file."""
    print(f"Loading episode from {episode_path.name}...")
    
    try:
        data = torch.load(episode_path)
        print(f"✓ Loaded episode with {len(data['frame_index'])} frames")
        return data
    except Exception as e:
        raise ValueError(f"Failed to load episode: {e}")


def find_all_episodes(dataset_dir: Path):
    """Find all episode files in a directory."""
    episode_files = sorted(dataset_dir.glob("episode_*.pt"))
    
    if len(episode_files) == 0:
        raise ValueError(f"No episode files found in {dataset_dir}")
    
    print(f"Found {len(episode_files)} episodes")
    return episode_files


def get_data_keys(data: dict):
    """
    Detect format and return appropriate keys.
    
    Supports both old and new LeRobot formats:
    - Old: 'observation.state', 'action', 'observation.images.*'
    - New: 'state', 'actions', 'image', 'wrist_image'
    """
    # Detect format by checking which keys exist
    if 'observation.state' in data:
        # Old format
        state_key = 'observation.state'
        action_key = 'action'
        camera_keys = [k for k in data.keys() if k.startswith('observation.images.')]
        camera_names = [k.replace('observation.images.', '') for k in camera_keys]
    else:
        # New format
        state_key = 'state'
        action_key = 'actions'
        camera_keys = []
        camera_names = []
        if 'image' in data:
            camera_keys.append('image')
            camera_names.append('image')
        if 'wrist_image' in data:
            camera_keys.append('wrist_image')
            camera_names.append('wrist_image')
    
    return state_key, action_key, camera_keys, camera_names


def visualize_with_rerun(
    data: dict,
    episode_name: str,
    fps: int = 30,
    save_path: Optional[Path] = None,
):
    """
    Visualize LeRobot episode using Rerun.
    
    Supports both formats:
    - Old: {'observation.state', 'action', 'observation.images.*'}
    - New: {'state', 'actions', 'image', 'wrist_image'}
    """
    # Initialize Rerun
    if save_path:
        rr.init(episode_name, recording_id=episode_name, spawn=False)
        rr.save(str(save_path))
        print(f"Recording will be saved to: {save_path}")
    else:
        rr.init(episode_name, recording_id=episode_name, spawn=True)
    
    # Log static metadata
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    
    # Detect format and get keys
    state_key, action_key, camera_keys, camera_names = get_data_keys(data)
    print(f"Detected format: state='{state_key}', action='{action_key}', cameras={camera_names}")
    
    # Extract data
    states = data[state_key].numpy()
    actions = data[action_key].numpy()
    timestamps = data.get('timestamp', None)
    if timestamps is not None:
        timestamps = timestamps.numpy()
    
    num_frames = len(states)
    
    print(f"\nVisualizing {num_frames} frames at {fps} FPS...")
    print(f"Episode index: {data['episode_index'][0].item()}")
    print(f"State dimensions: {states.shape}")
    print(f"Action dimensions: {actions.shape}")
    print(f"Camera views: {camera_names}")
    print("=" * 60)
    
    # Process each frame
    for frame_idx in range(num_frames):
        # Set timeline
        if timestamps is not None:
            time_sec = float(timestamps[frame_idx])
        else:
            time_sec = frame_idx / fps
        
        rr.set_time_seconds("timestamp", time_sec)
        rr.set_time_sequence("frame", frame_idx)
        
        # === Robot State ===
        state = states[frame_idx]
        
        # Log individual joint positions as scalars
        for joint_idx, pos in enumerate(state):
            rr.log(
                f"robot/state/joint_{joint_idx}",
                rr.Scalars(float(pos)),
            )
        
        # Log all joints as tensor (for plotting together)
        rr.log("robot/state", rr.Tensor(state))
        
        # === Actions ===
        action = actions[frame_idx]
        
        # Log individual actions
        for joint_idx, cmd in enumerate(action):
            rr.log(
                f"robot/action/joint_{joint_idx}",
                rr.Scalars(float(cmd)),
            )
        
        # Log all actions as tensor
        rr.log("robot/action", rr.Tensor(action))
        
        # === Tracking Error ===
        if state.shape == action.shape:
            tracking_error = np.abs(state - action)
            
            # Log individual tracking errors
            for joint_idx, err in enumerate(tracking_error):
                rr.log(
                    f"robot/tracking_error/joint_{joint_idx}",
                    rr.Scalars(float(err)),
                )
            
            # Log mean tracking error
            rr.log("robot/tracking_error/mean", rr.Scalars(float(np.mean(tracking_error))))
            rr.log("robot/tracking_error/max", rr.Scalars(float(np.max(tracking_error))))
        
        # === Camera Images ===
        for camera_key, camera_name in zip(camera_keys, camera_names):
            camera_data = data[camera_key][frame_idx]
            
            # Handle both CHW (PyTorch fallback) and HWC (official LeRobot) formats
            if len(camera_data.shape) == 3:
                if camera_data.shape[0] in [1, 3]:  # CHW format: (C, H, W)
                    image = camera_data.numpy().transpose(1, 2, 0)  # Convert to HWC
                else:  # HWC format: (H, W, C)
                    image = camera_data.numpy()
            else:
                image = camera_data.numpy()
            
            # Ensure uint8 for proper display
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Log RGB image
            if image.shape[-1] == 3:
                rr.log(f"cameras/{camera_name}/rgb", rr.Image(image))
            elif image.shape[-1] == 1:
                # Single channel (could be depth or grayscale)
                rr.log(f"cameras/{camera_name}/depth", rr.DepthImage(image.squeeze()))
        
        # === Episode Progress ===
        progress = (frame_idx + 1) / num_frames
        rr.log("episode/progress", rr.Scalars(progress))
        
        # === Done Flag ===
        is_done = data['next.done'][frame_idx].item()
        rr.log("episode/done", rr.Scalars(1.0 if is_done else 0.0))
        
        # Progress indicator
        if frame_idx % 10 == 0 or frame_idx == num_frames - 1:
            progress_pct = (frame_idx + 1) / num_frames * 100
            print(f"Progress: {progress_pct:.1f}% ({frame_idx + 1}/{num_frames} frames)", end='\r')
    
    print(f"\n{'=' * 60}")
    print(f"✓ Visualization complete!")
    print(f"  Total frames: {num_frames}")
    print(f"  Duration: {num_frames / fps:.2f} seconds")
    
    if save_path:
        print(f"  Saved to: {save_path}")
        print(f"\nView with: rerun {save_path}")
    else:
        print("\nRerun viewer should be open.")
        print("Use the timeline at the bottom to scrub through frames!")


def visualize_multiple_episodes(
    episode_files: list,
    fps: int = 30,
    save_path: Optional[Path] = None,
):
    """Visualize multiple episodes in sequence."""
    # Initialize Rerun once for all episodes
    recording_name = "lerobot_dataset"
    
    if save_path:
        rr.init(recording_name, recording_id=recording_name, spawn=False)
        rr.save(str(save_path))
        print(f"Recording will be saved to: {save_path}")
    else:
        rr.init(recording_name, recording_id=recording_name, spawn=True)
    
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    
    print(f"\nVisualizing {len(episode_files)} episodes...")
    print("=" * 60)
    
    total_frames = 0
    
    for ep_idx, episode_file in enumerate(episode_files):
        print(f"\nEpisode {ep_idx + 1}/{len(episode_files)}: {episode_file.name}")
        
        # Load episode
        data = load_lerobot_episode(episode_file)
        
        # Detect format and get keys
        state_key, action_key, camera_keys, camera_names = get_data_keys(data)
        
        states = data[state_key].numpy()
        actions = data[action_key].numpy()
        timestamps = data.get('timestamp', None)
        if timestamps is not None:
            timestamps = timestamps.numpy()
        
        num_frames = len(states)
        
        # Process each frame
        for frame_idx in range(num_frames):
            global_frame_idx = total_frames + frame_idx
            
            # Set timeline
            if timestamps is not None:
                time_sec = float(timestamps[frame_idx])
            else:
                time_sec = global_frame_idx / fps
            
            rr.set_time_seconds("timestamp", time_sec)
            rr.set_time_sequence("frame", global_frame_idx)
            rr.set_time_sequence("episode", ep_idx)
            
            # Log episode info
            rr.log("episode/index", rr.Scalars(float(ep_idx)))
            
            # Robot state
            state = states[frame_idx]
            for joint_idx, pos in enumerate(state):
                rr.log(f"robot/state/joint_{joint_idx}", rr.Scalars(float(pos)))
            rr.log("robot/state", rr.Tensor(state))
            
            # Actions
            action = actions[frame_idx]
            for joint_idx, cmd in enumerate(action):
                rr.log(f"robot/action/joint_{joint_idx}", rr.Scalars(float(cmd)))
            rr.log("robot/action", rr.Tensor(action))
            
            # Tracking error
            if state.shape == action.shape:
                tracking_error = np.abs(state - action)
                for joint_idx, err in enumerate(tracking_error):
                    rr.log(f"robot/tracking_error/joint_{joint_idx}", rr.Scalars(float(err)))
            
            # Camera images
            for camera_key, camera_name in zip(camera_keys, camera_names):
                camera_data = data[camera_key][frame_idx]
                
                # Handle both CHW and HWC formats
                if len(camera_data.shape) == 3:
                    if camera_data.shape[0] in [1, 3]:  # CHW format
                        image = camera_data.numpy().transpose(1, 2, 0)
                    else:  # HWC format
                        image = camera_data.numpy()
                else:
                    image = camera_data.numpy()
                
                # Ensure uint8
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                
                if image.shape[-1] == 3:
                    rr.log(f"cameras/{camera_name}/rgb", rr.Image(image))
                elif image.shape[-1] == 1:
                    rr.log(f"cameras/{camera_name}/depth", rr.DepthImage(image.squeeze()))
            
            # Progress
            is_done = data['next.done'][frame_idx].item()
            rr.log("episode/done", rr.Scalars(1.0 if is_done else 0.0))
        
        total_frames += num_frames
        print(f"  ✓ Processed {num_frames} frames (total: {total_frames})")
    
    print(f"\n{'=' * 60}")
    print(f"✓ All episodes visualized!")
    print(f"  Total episodes: {len(episode_files)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames / fps:.2f} seconds")
    
    if save_path:
        print(f"  Saved to: {save_path}")
        print(f"\nView with: rerun {save_path}")
    else:
        print("\nRerun viewer should be open.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize converted LeRobot dataset using Rerun"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to episode .pt file or directory containing episodes",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for timeline (default: 30)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save recording to .rrd file instead of opening viewer",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Episode name (defaults to file/directory name)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Visualize all episodes in directory",
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"Error: Path not found: {path}")
        return
    
    save_path = Path(args.save) if args.save else None
    
    # Check if path is a directory or file
    if path.is_dir():
        # Directory - find all episodes
        try:
            episode_files = find_all_episodes(path)
        except ValueError as e:
            print(f"Error: {e}")
            return
        
        if args.all or len(episode_files) > 1:
            # Visualize all episodes
            visualize_multiple_episodes(episode_files, args.fps, save_path)
        else:
            # Just visualize the first episode
            data = load_lerobot_episode(episode_files[0])
            episode_name = args.name or episode_files[0].stem
            visualize_with_rerun(data, episode_name, args.fps, save_path)
    
    elif path.is_file():
        # Single episode file
        if not path.suffix == '.pt':
            print(f"Error: Expected .pt file, got {path.suffix}")
            return
        
        # Load and visualize
        data = load_lerobot_episode(path)
        
        # Print episode info
        print("\nEpisode data:")
        for key, value in data.items():
            if hasattr(value, 'shape'):
                print(f"  {key:30s}: {value.shape} {value.dtype}")
            else:
                print(f"  {key:30s}: {type(value)}")
        
        episode_name = args.name or path.stem
        visualize_with_rerun(data, episode_name, args.fps, save_path)
    
    else:
        print(f"Error: Path is neither file nor directory: {path}")


if __name__ == "__main__":
    main()

