#!/usr/bin/env python3
"""
Visualize GELLO episode data using Rerun.

Rerun provides an interactive 3D viewer with timeline scrubbing, perfect for robot data.

Installation:
    pip install rerun-sdk

Usage:
    # Visualize a single episode
    python scripts/visualize_episode_rerun.py data/GelloAgent/1028_143052

    # Save recording for later viewing
    python scripts/visualize_episode_rerun.py data/GelloAgent/1028_143052 --save episode.rrd

    # View saved recording
    rerun episode.rrd
"""

import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import rerun as rr


def load_gello_episode(episode_dir: Path):
    """Load all pickle files from an episode directory."""
    pkl_files = sorted(episode_dir.glob("*.pkl"))
    
    if len(pkl_files) == 0:
        raise ValueError(f"No pickle files found in {episode_dir}")
    
    print(f"Loading {len(pkl_files)} frames from {episode_dir.name}...")
    
    frames = []
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, "rb") as f:
                frame = pickle.load(f)
                frames.append(frame)
        except Exception as e:
            print(f"Warning: Failed to load {pkl_file}: {e}")
            continue
    
    print(f"✓ Loaded {len(frames)} frames")
    return frames


def visualize_with_rerun(
    frames: list,
    episode_name: str,
    fps: int = 30,
    save_path: Optional[Path] = None,
):
    """
    Visualize episode using Rerun.
    
    Creates an interactive timeline with:
    - Joint positions plot
    - Actions plot
    - Camera images (wrist + side)
    - 3D robot visualization (if available)
    """
    # Initialize Rerun
    if save_path:
        rr.init(episode_name, recording_id=episode_name, spawn=False)
        rr.save(str(save_path))
        print(f"Recording will be saved to: {save_path}")
    else:
        rr.init(episode_name, recording_id=episode_name, spawn=True)
    
    # Log static metadata
    rr.log("world", rr.ViewCoordinates.RDF, static=True)  # Robot: X=Right, Y=Down, Z=Forward
    
    print(f"\nVisualizing {len(frames)} frames at {fps} FPS...")
    print("=" * 60)
    
    # Process each frame
    for frame_idx, frame in enumerate(frames):
        # Set timeline
        time_sec = frame_idx / fps
        rr.set_time_seconds("timestamp", time_sec)
        rr.set_time_sequence("frame", frame_idx)
        
        # === Robot State ===
        joint_positions = frame.get('joint_positions')
        if joint_positions is not None:
            # Log individual joint positions as scalar timeline
            for joint_idx, pos in enumerate(joint_positions):
                rr.log(
                    f"robot/joints/joint_{joint_idx}",
                    rr.Scalars(float(pos)),
                )
            
            # Log all joints as a tensor (for plotting together)
            rr.log("robot/joint_positions", rr.Tensor(joint_positions))
        
        # === Joint Velocities ===
        joint_velocities = frame.get('joint_velocities')
        if joint_velocities is not None:
            rr.log("robot/joint_velocities", rr.Tensor(joint_velocities))
        
        # === Actions/Control Commands ===
        control = frame.get('control')
        if control is not None:
            # Log individual control commands
            for joint_idx, cmd in enumerate(control):
                rr.log(
                    f"robot/control/joint_{joint_idx}",
                    rr.Scalars(float(cmd)),
                )
            
            # Log tracking error (difference between state and command)
            if joint_positions is not None:
                tracking_error = np.abs(joint_positions - control)
                for joint_idx, err in enumerate(tracking_error):
                    rr.log(
                        f"robot/tracking_error/joint_{joint_idx}",
                        rr.Scalars(float(err)),
                    )
        
        # === End-Effector Pose ===
        ee_pos_quat = frame.get('ee_pos_quat')
        if ee_pos_quat is not None and len(ee_pos_quat) == 7:
            # Position (first 3 elements)
            position = ee_pos_quat[:3]
            # Quaternion (last 4 elements: w, x, y, z)
            quaternion = ee_pos_quat[3:]  # Assuming [w, x, y, z] format
            
            # Log as 3D transform
            rr.log(
                "robot/end_effector",
                rr.Transform3D(
                    translation=position,
                    rotation=rr.Quaternion(xyzw=[quaternion[1], quaternion[2], quaternion[3], quaternion[0]]),
                ),
            )
            
            # Log position as point cloud for visualization
            rr.log(
                "robot/end_effector/position",
                rr.Points3D(positions=[position], radii=0.02, colors=[255, 0, 0]),
            )
        
        # === Gripper State ===
        gripper_pos = frame.get('gripper_position')
        if gripper_pos is not None:
            gripper_val = gripper_pos if np.isscalar(gripper_pos) else gripper_pos[0]
            rr.log("robot/gripper_position", rr.Scalars(float(gripper_val)))
        
        # === Camera Images ===
        # Wrist camera
        wrist_rgb = frame.get('wrist_image')
        if wrist_rgb is not None:
            rr.log("cameras/wrist_image/rgb", rr.Image(wrist_rgb))
        
        wrist_depth = frame.get('wrist_image_depth')
        if wrist_depth is not None:
            # Normalize depth for visualization
            depth_normalized = wrist_depth.squeeze()
            rr.log("cameras/wrist_image/depth", rr.DepthImage(depth_normalized))
        
        # Side camera
        side_rgb = frame.get('image')
        if side_rgb is not None:
            rr.log("cameras/image/rgb", rr.Image(side_rgb))
        
        side_depth = frame.get('image_depth')
        if side_depth is not None:
            depth_normalized = side_depth.squeeze()
            rr.log("cameras/image/depth", rr.DepthImage(depth_normalized))
        
        # Progress indicator
        if frame_idx % 10 == 0:
            progress = (frame_idx + 1) / len(frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_idx + 1}/{len(frames)} frames)", end='\r')
    
    print(f"\n{'=' * 60}")
    print(f"✓ Visualization complete!")
    print(f"  Total frames: {len(frames)}")
    print(f"  Duration: {len(frames) / fps:.2f} seconds")
    
    if save_path:
        print(f"  Saved to: {save_path}")
        print(f"\nView with: rerun {save_path}")
    else:
        print("\nRerun viewer should be open.")
        print("Use the timeline at the bottom to scrub through frames!")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GELLO episode data using Rerun"
    )
    parser.add_argument(
        "episode_dir",
        type=str,
        help="Directory containing episode pickle files",
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
        help="Episode name (defaults to directory name)",
    )
    
    args = parser.parse_args()
    
    episode_dir = Path(args.episode_dir)
    
    if not episode_dir.exists():
        print(f"Error: Directory not found: {episode_dir}")
        return
    
    if not episode_dir.is_dir():
        print(f"Error: Not a directory: {episode_dir}")
        return
    
    # Load episode
    frames = load_gello_episode(episode_dir)
    
    if len(frames) == 0:
        print("Error: No frames loaded!")
        return
    
    # Print episode info
    print("\nEpisode info:")
    print(f"  Directory: {episode_dir}")
    print(f"  Frames: {len(frames)}")
    print(f"  Duration: {len(frames) / args.fps:.2f} seconds")
    
    # Check what data is available
    sample_frame = frames[0]
    print("\nAvailable data:")
    for key in sample_frame.keys():
        value = sample_frame[key]
        if isinstance(value, np.ndarray):
            print(f"  {key:20s}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key:20s}: {type(value)}")
    
    # Determine episode name
    episode_name = args.name or episode_dir.name
    
    # Visualize
    save_path = Path(args.save) if args.save else None
    visualize_with_rerun(frames, episode_name, args.fps, save_path)


if __name__ == "__main__":
    main()

