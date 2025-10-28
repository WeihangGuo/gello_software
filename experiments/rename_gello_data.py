#!/usr/bin/env python3
"""
Rename camera keys in GELLO pickle files.

Transforms:
    'wrist_rgb' → 'image'
    'wrist_depth' → 'image_depth'
    'side_rgb' → 'wrist_image'
    'side_depth' → 'wrist_image_depth'

This is useful for matching training framework conventions where:
- 'image' is the main third-person view
- 'wrist_image' is the wrist-mounted camera

Usage:
    # Rename in-place (creates backup)
    python experiments/rename_gello_data.py \
        --input-dir data/GelloAgent/1028_143052

    # Save to new directory
    python experiments/rename_gello_data.py \
        --input-dir data/GelloAgent/1028_143052 \
        --output-dir data/GelloAgent/1028_143052_renamed

    # Process multiple episodes recursively
    python experiments/rename_gello_data.py \
        --input-dir data/GelloAgent \
        --output-dir data/GelloAgent_renamed \
        --recursive
"""

import argparse
import pickle
import shutil
from pathlib import Path
from typing import Dict, Any

from tqdm import tqdm


def rename_camera_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rename camera keys in a GELLO data frame.
    
    Args:
        data: Original GELLO data dictionary
        
    Returns:
        New dictionary with renamed keys
    """
    renamed_data = {}
    
    # Mapping: old_key → new_key
    key_mapping = {
        'wrist_rgb': 'image',
        'wrist_depth': 'image_depth',
        'side_rgb': 'wrist_image',
        'side_depth': 'wrist_image_depth',
    }
    
    for old_key, value in data.items():
        # Rename if key is in mapping, otherwise keep original
        new_key = key_mapping.get(old_key, old_key)
        renamed_data[new_key] = value
    
    return renamed_data


def process_pickle_file(input_path: Path, output_path: Path, dry_run: bool = False):
    """Process a single pickle file."""
    try:
        # Load original data
        with open(input_path, 'rb') as f:
            original_data = pickle.load(f)
        
        # Rename keys
        renamed_data = rename_camera_keys(original_data)
        
        # Save to output
        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(renamed_data, f)
        
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def find_pickle_files(input_dir: Path, recursive: bool = False) -> Dict[Path, Path]:
    """
    Find all pickle files and determine output paths.
    
    Returns:
        Dict mapping input_path → relative_path (for organizing output)
    """
    files = {}
    
    if recursive:
        # Find all pickle files in subdirectories
        for pkl_file in input_dir.rglob("*.pkl"):
            # Get relative path from input_dir
            rel_path = pkl_file.relative_to(input_dir)
            files[pkl_file] = rel_path
    else:
        # Single directory
        for pkl_file in input_dir.glob("*.pkl"):
            files[pkl_file] = pkl_file.name
    
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Rename camera keys in GELLO pickle files"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing GELLO pickle files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (if not specified, renames in-place with backup)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process all subdirectories recursively",
    )
    parser.add_argument(
        "--backup-suffix",
        type=str,
        default=".backup",
        help="Backup suffix when renaming in-place (default: .backup)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually modifying files",
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Determine output mode
    in_place = args.output_dir is None
    if in_place:
        output_dir = input_dir
        print("Mode: In-place rename (will create backups)")
    else:
        output_dir = Path(args.output_dir)
        print(f"Mode: Copy to new directory")
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GELLO Camera Key Renamer")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Recursive: {args.recursive}")
    print(f"Dry run: {args.dry_run}")
    print("\nKey mapping:")
    print("  'wrist_rgb'   → 'image'")
    print("  'wrist_depth' → 'image_depth'")
    print("  'side_rgb'    → 'wrist_image'")
    print("  'side_depth'  → 'wrist_image_depth'")
    print("=" * 60)
    
    # Find all pickle files
    pickle_files = find_pickle_files(input_dir, args.recursive)
    
    if len(pickle_files) == 0:
        print("No pickle files found!")
        return
    
    print(f"\nFound {len(pickle_files)} pickle files")
    
    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for input_path, rel_path in list(pickle_files.items())[:10]:
            output_path = output_dir / rel_path
            print(f"  {input_path} → {output_path}")
        if len(pickle_files) > 10:
            print(f"  ... and {len(pickle_files) - 10} more files")
        return
    
    # Confirm if in-place and not dry-run
    if in_place and not args.dry_run:
        print(f"\n⚠️  WARNING: This will modify files in {input_dir}")
        print(f"   Backups will be created with suffix '{args.backup_suffix}'")
        response = input("\nContinue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Process files
    success_count = 0
    
    for input_path, rel_path in tqdm(pickle_files.items(), desc="Processing files"):
        # Determine output path
        if in_place:
            # Create backup
            backup_path = input_path.with_suffix(input_path.suffix + args.backup_suffix)
            if not backup_path.exists():
                shutil.copy2(input_path, backup_path)
            output_path = input_path
        else:
            output_path = output_dir / rel_path
        
        # Process file
        if process_pickle_file(input_path, output_path, args.dry_run):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✓ Successfully processed {success_count}/{len(pickle_files)} files")
    
    if in_place:
        print(f"   Original files backed up with suffix '{args.backup_suffix}'")
    else:
        print(f"   Renamed files saved to: {output_dir}")
    
    print("=" * 60)
    
    # Show example of changes
    if success_count > 0:
        print("\nVerify changes:")
        first_file = list(pickle_files.keys())[0]
        if in_place:
            verify_file = first_file
        else:
            verify_file = output_dir / pickle_files[first_file]
        
        print(f"\npython -c \"")
        print(f"import pickle")
        print(f"with open('{verify_file}', 'rb') as f:")
        print(f"    data = pickle.load(f)")
        print(f"print('Keys:', list(data.keys()))")
        print(f"\"")


if __name__ == "__main__":
    main()

