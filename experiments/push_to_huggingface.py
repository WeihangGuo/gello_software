#!/usr/bin/env python3
"""
Push dataset to Hugging Face Hub.

Usage:
    python experiments/push_to_huggingface.py \
        --dataset-dir lerobot_datasets/pick_up_yellow_lego \
        --repo-id weihang44/kavraki.ur5s \
        --task-name pick_up_yellow_lego
        
Or push multiple datasets:
    python experiments/push_to_huggingface.py \
        --dataset-dirs lerobot_datasets/pick_up_yellow_lego lerobot_datasets/pick_up_green_lego \
        --repo-id weihang44/kavraki.ur5s
USED!!
    python experiments/push_to_huggingface.py --dataset-dirs lerobot_datasets/pick_up_yellow_lego lerobot_datasets/pick_up_green_lego --repo-id weihang44/kavraki.ur5s
"""

import argparse
import torch
from pathlib import Path
from typing import List, Optional
from huggingface_hub import HfApi, create_repo


def push_dataset_to_hub(
    dataset_dir: Path,
    repo_id: str,
    task_name: Optional[str] = None,
    private: bool = False,
):
    """
    Push a dataset directory to Hugging Face Hub.
    
    Args:
        dataset_dir: Path to the dataset directory containing episode_*.pt files
        repo_id: Repository ID on Hugging Face (e.g., 'username/dataset-name')
        task_name: Optional task name to organize datasets (will create subdirectory)
        private: Whether to create a private repository
    """
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    
    # Check if dataset has episodes
    episode_files = sorted(dataset_dir.glob("episode_*.pt"))
    metadata_file = dataset_dir / "metadata.pt"
    
    if not episode_files:
        raise ValueError(f"No episode files found in {dataset_dir}")
    
    print(f"Found {len(episode_files)} episodes in {dataset_dir}")
    
    # Load metadata if available
    if metadata_file.exists():
        metadata = torch.load(metadata_file)
        print(f"Metadata: {metadata}")
    
    # Load first episode to check structure
    first_ep = torch.load(episode_files[0])
    print(f"Episode structure:")
    for key, value in first_ep.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    # Create repository if it doesn't exist
    print(f"\nCreating/accessing repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        print(f"✓ Repository ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Warning: Could not create repository: {e}")
        print("If the repository already exists, continuing...")
    
    # Upload files
    api = HfApi()
    
    # Determine upload path
    if task_name:
        path_in_repo = f"{task_name}/"
        print(f"\nUploading files to {repo_id}/{task_name}/")
    else:
        # Use the dataset directory name as task name
        task_name = dataset_dir.name
        path_in_repo = f"{task_name}/"
        print(f"\nUploading files to {repo_id}/{task_name}/")
    
    # Upload all episode files and metadata
    files_to_upload = episode_files + ([metadata_file] if metadata_file.exists() else [])
    
    print(f"Uploading {len(files_to_upload)} files...")
    for file_path in files_to_upload:
        print(f"  Uploading {file_path.name}...")
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=f"{path_in_repo}{file_path.name}",
            repo_id=repo_id,
            repo_type="dataset",
        )
    
    print(f"✓ Successfully uploaded {len(files_to_upload)} files to {repo_id}")
    print(f"\nView your dataset at: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Push dataset(s) to Hugging Face Hub"
    )
    
    # Single dataset mode
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Path to a single dataset directory",
    )
    
    # Multiple datasets mode
    parser.add_argument(
        "--dataset-dirs",
        type=str,
        nargs="+",
        help="Paths to multiple dataset directories",
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., 'username/dataset-name')",
    )
    
    parser.add_argument(
        "--task-name",
        type=str,
        help="Optional task name (used when pushing a single dataset)",
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )
    
    args = parser.parse_args()
    
    # Collect dataset directories
    dataset_dirs: List[Path] = []
    
    if args.dataset_dir:
        dataset_dirs.append(Path(args.dataset_dir))
    
    if args.dataset_dirs:
        dataset_dirs.extend([Path(d) for d in args.dataset_dirs])
    
    if not dataset_dirs:
        parser.error("Must provide --dataset-dir or --dataset-dirs")
    
    print("=" * 70)
    print("Push Dataset to Hugging Face Hub")
    print("=" * 70)
    print(f"Repository: {args.repo_id}")
    print(f"Datasets: {len(dataset_dirs)}")
    for d in dataset_dirs:
        print(f"  - {d}")
    print(f"Private: {args.private}")
    print("=" * 70)
    
    # Push each dataset
    for dataset_dir in dataset_dirs:
        print(f"\n{'=' * 70}")
        print(f"Processing: {dataset_dir}")
        print('=' * 70)
        
        try:
            # Use provided task name or derive from directory name
            task_name = args.task_name if len(dataset_dirs) == 1 and args.task_name else None
            
            push_dataset_to_hub(
                dataset_dir=dataset_dir,
                repo_id=args.repo_id,
                task_name=task_name,
                private=args.private,
            )
        except Exception as e:
            print(f"Error pushing {dataset_dir}: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("Upload complete!")
    print(f"View your dataset at: https://huggingface.co/datasets/{args.repo_id}")
    print("=" * 70)


if __name__ == "__main__":
    main()

