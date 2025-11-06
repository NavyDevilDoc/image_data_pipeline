import shutil
import random
import json
from datetime import datetime
from pathlib import Path
import yaml
from config import config, get_project_title

def create_yolo_dataset_structure(base_dir, train_ratio=0.8, val_ratio=0.2, seed=42, 
                                  images_subdir=None, labels_subdir=None):
    """
    Split annotated dataset into YOLO training structure
    
    Args:
        base_dir: Path to your poop_detection_dataset directory
        train_ratio: Fraction for training (0.8 = 80%)
        val_ratio: Fraction for validation (0.2 = 20%)
        seed: Random seed for reproducible splits
        images_subdir: Override images directory (e.g., 'augmented_images')
        labels_subdir: Override labels directory (e.g., 'augmented_labels')
    
    Returns:
        Dictionary with dataset statistics
    """
    
    random.seed(seed)
    base_path = Path(base_dir)
    
    # Auto-detect available datasets or use specified ones
    if images_subdir is None or labels_subdir is None:
        available_datasets = detect_available_datasets(base_path)
        
        if len(available_datasets) > 1:
            print("📁 Multiple datasets detected:")
            for i, (name, img_dir, lbl_dir, count) in enumerate(available_datasets, 1):
                print(f"   {i}. {name}: {count} images")
            
            choice = input(f"Select dataset (1-{len(available_datasets)}, default: 1): ").strip()
            choice_idx = int(choice) - 1 if choice.isdigit() else 0
            
            if 0 <= choice_idx < len(available_datasets):
                _, images_dir, labels_dir, _ = available_datasets[choice_idx]
            else:
                print("Invalid choice, using first dataset")
                _, images_dir, labels_dir, _ = available_datasets[0]
        else:
            # Only one dataset available
            _, images_dir, labels_dir, _ = available_datasets[0]
            print(f"📁 Using dataset: {images_dir.name}")
    else:
        # Use specified directories
        images_dir = base_path / images_subdir
        labels_dir = base_path / labels_subdir
    
    print(f"📊 Source directories:")
    print(f"   • Images: {images_dir}")
    print(f"   • Labels: {labels_dir}")
    
    # Create output structure
    train_dir = base_path / 'train'
    val_dir = base_path / 'val'
    
    # Create directories
    for split_dir in [train_dir, val_dir]:
        (split_dir / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(images_dir.glob('*.jpg'))
    
    # Validate we have both images and labels
    valid_pairs = []
    for img_file in image_files:
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            valid_pairs.append(img_file.stem)
        else:
            print(f"⚠️  No label found for {img_file.name}")
    
    print(f"📊 Found {len(valid_pairs)} valid image-label pairs")
    
    if len(valid_pairs) == 0:
        print("❌ No valid image-label pairs found!")
        return None
    
    # Calculate total samples for ratio calculations
    total_samples = len(valid_pairs)
    
    # ✅ NEW: Check for existing training history to preserve validation set
    training_history = load_training_history(base_path)
    latest_model = max(training_history.keys()) if training_history else None
    
    if latest_model and 'validation_set' in training_history[latest_model]:
        # Use existing validation set for consistency
        previous_val_set = training_history[latest_model]['validation_set']
        
        # Filter previous validation set to only include currently available images
        current_val = [img for img in previous_val_set if img in valid_pairs]
        current_train = [img for img in valid_pairs if img not in current_val]
        
        print(f"📊 Using preserved validation set from {latest_model}")
        print(f"   • Validation images: {len(current_val)} (preserved)")
        print(f"   • Training images: {len(current_train)} (including any new images)")
        
        # If validation set is too small due to missing images, supplement it
        if len(current_val) < total_samples * val_ratio * 0.8:  # If < 80% of target val size
            print("⚠️ Previous validation set too small, supplementing with new images...")
            additional_needed = int(total_samples * val_ratio) - len(current_val)
            
            # Randomly select additional validation images from training pool
            random.shuffle(current_train)
            additional_val = current_train[:additional_needed]
            current_train = current_train[additional_needed:]
            current_val.extend(additional_val)
            
            print(f"   • Added {len(additional_val)} new images to validation set")
    
    else:
        # First time training - create new split
        print("📊 Creating initial train/validation split")
        random.shuffle(valid_pairs)
        train_count = int(total_samples * train_ratio)
        
        current_train = valid_pairs[:train_count]
        current_val = valid_pairs[train_count:]
        
        print(f"   • Training images: {len(current_train)}")
        print(f"   • Validation images: {len(current_val)}")
    
    # Determine model version for history tracking
    model_version = f"v{len(training_history) + 1}"
    
    # Copy files to respective directories
    print(f"📁 Copying {len(current_train)} files to training set...")
    for file_stem in current_train:
        # Copy image
        src_img = images_dir / f"{file_stem}.jpg"
        dst_img = train_dir / 'images' / f"{file_stem}.jpg"
        shutil.copy2(src_img, dst_img)
        
        # Copy label
        src_lbl = labels_dir / f"{file_stem}.txt"
        dst_lbl = train_dir / 'labels' / f"{file_stem}.txt"
        shutil.copy2(src_lbl, dst_lbl)
    
    print(f"📁 Copying {len(current_val)} files to validation set...")
    for file_stem in current_val:
        # Copy image
        src_img = images_dir / f"{file_stem}.jpg"
        dst_img = val_dir / 'images' / f"{file_stem}.jpg"
        shutil.copy2(src_img, dst_img)
        
        # Copy label
        src_lbl = labels_dir / f"{file_stem}.txt"
        dst_lbl = val_dir / 'labels' / f"{file_stem}.txt"
        shutil.copy2(src_lbl, dst_lbl)
    
    # Read existing class information from data.yaml (created by yolo_conversion.py)
    yaml_path = base_path / 'data.yaml'
    
    if yaml_path.exists():
        print("📄 Using existing data.yaml configuration")
        
        # Just verify it has the right paths
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Update only the path if needed
        if str(base_path.absolute()) != config_data.get('path'):
            config_data['path'] = str(base_path.absolute())
            with open(yaml_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            print(f"📄 Updated path in data.yaml")
        
        nc = config_data.get('nc', 1)
        names = config_data.get('names', ['poop'])
        print(f"   • Classes: {nc}")
        print(f"   • Names: {names}")
        
    else:
        print("❌ No data.yaml found! Run yolo_conversion.py first.")
        return None
    
    # Create YOLO data.yaml configuration file with correct class info
    data_yaml = {
        'path': str(base_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': nc,
        'names': names
    }
    
    yaml_path = base_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"📄 Updated YOLO config: {yaml_path}")
    print(f"   • Classes: {nc}")
    print(f"   • Names: {names}")
    
    # ✅ NEW: Save training history for future incremental learning
    save_training_history(base_path, model_version, current_train, current_val, total_samples)
    
    # Return statistics
    stats = {
        'total_samples': total_samples,
        'train_samples': len(current_train),
        'val_samples': len(current_val),
        'train_ratio': len(current_train) / total_samples,
        'val_ratio': len(current_val) / total_samples,
        'yaml_config': str(yaml_path),
        'model_version': model_version 
    }
    
    return stats

def detect_available_datasets(base_path):
    """
    Detect all available image/label dataset combinations
    
    Args:
        base_path: Base dataset directory
        
    Returns:
        List of tuples: (name, images_dir, labels_dir, image_count)
    """
    
    datasets = []
    
    # Check for original dataset
    orig_images = base_path / 'images'
    orig_labels = base_path / 'yolo_labels'
    
    if orig_images.exists() and orig_labels.exists():
        img_count = len(list(orig_images.glob('*.jpg')))
        if img_count > 0:
            datasets.append(("Original Dataset", orig_images, orig_labels, img_count))
    
    # Check for augmented dataset
    aug_images = base_path / 'augmented_images'
    aug_labels = base_path / 'augmented_labels'
    
    if aug_images.exists() and aug_labels.exists():
        img_count = len(list(aug_images.glob('*.jpg')))
        if img_count > 0:
            datasets.append(("Augmented Dataset", aug_images, aug_labels, img_count))
    
    # Check for any other potential datasets
    for images_dir in base_path.glob('*images*'):
        if images_dir.is_dir() and images_dir not in [orig_images, aug_images]:
            # Look for corresponding labels directory
            possible_label_dirs = [
                base_path / images_dir.name.replace('images', 'labels'),
                base_path / f"{images_dir.stem}_labels",
                base_path / 'yolo_labels'
            ]
            
            for labels_dir in possible_label_dirs:
                if labels_dir.exists():
                    img_count = len(list(images_dir.glob('*.jpg')))
                    if img_count > 0:
                        datasets.append((f"Custom Dataset ({images_dir.name})", 
                                       images_dir, labels_dir, img_count))
                    break
    
    return datasets


def load_training_history(base_path):
    """Load training history to preserve validation set consistency"""
    history_file = base_path / '.training_history.json'
    
    if history_file.exists():
        with open(history_file, 'r') as f:
            return json.load(f)
    else:
        return {}

def save_training_history(base_path, model_version, train_files, val_files, total_images):
    """Save training history for future reference"""
    history_file = base_path / '.training_history.json'
    
    # Load existing history
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = {}
    
    # Add current training cycle
    history[model_version] = {
        'training_date': datetime.now().isoformat(),
        'total_images': total_images,
        'train_images': len(train_files),
        'val_images': len(val_files),
        'validation_set': val_files,  # Preserve exact validation set
        'dataset_used': 'augmented' if 'augmented' in str(base_path) else 'original'
    }
    
    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"📝 Training history saved: {model_version}")


def validate_dataset_structure(base_dir):
    """
    Validate the created dataset structure
    
    Args:
        base_dir: Path to dataset directory
        
    Returns:
        Validation report
    """
    
    base_path = Path(base_dir)
    
    # Check required directories exist
    required_dirs = [
        'train/images', 'train/labels',
        'val/images', 'val/labels'
    ]
    
    print("🔍 Validating dataset structure...")
    
    validation_report = {}
    
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        exists = full_path.exists()
        
        if exists:
            file_count = len(list(full_path.glob('*')))
            validation_report[dir_path] = {'exists': True, 'files': file_count}
            print(f"✅ {dir_path}: {file_count} files")
        else:
            validation_report[dir_path] = {'exists': False, 'files': 0}
            print(f"❌ {dir_path}: Not found")
    
    # Check data.yaml exists
    yaml_path = base_path / 'data.yaml'
    if yaml_path.exists():
        print(f"✅ data.yaml: Found")
        validation_report['data_yaml'] = True
    else:
        print(f"❌ data.yaml: Not found") 
        validation_report['data_yaml'] = False
    
    return validation_report

def print_dataset_summary(stats):
    """
    Print a nice summary of the dataset split
    
    Args:
        stats: Statistics dictionary from create_yolo_dataset_structure
    """
    
    print("\n" + "="*50)
    print("📊 DATASET SPLIT SUMMARY")
    print("="*50)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Training samples: {stats['train_samples']} ({stats['train_ratio']:.1%})")
    print(f"Validation samples: {stats['val_samples']} ({stats['val_ratio']:.1%})")
    print(f"Config file: {stats['yaml_config']}")
    print("="*50)
    print("✅ Ready for YOLO training!")


def get_initial_dataset_stats(dataset_dir):
    """Get initial dataset statistics for comparison"""
    try:
        dataset_path = Path(dataset_dir)
        
        # Check what datasets are available
        available_datasets = detect_available_datasets(dataset_path)
        
        # Check existing training history
        history = load_training_history(dataset_path)
        
        return {
            'available_datasets': len(available_datasets),
            'existing_models': len(history),
            'has_training_history': len(history) > 0
        }
    except:
        return {
            'available_datasets': 0,
            'existing_models': 0,
            'has_training_history': False
        }

def print_dataset_splitting_session_summary(initial_stats, final_stats, dataset_dir):
    """Generate user-friendly dataset splitting session summary with recommendations"""
    
    print("\n" + "="*60)
    print("📋 DATASET SPLITTING COMPLETED")
    print("="*60)
    
    # Session Results
    print(f"⏰ Session Results:")
    print(f"   • Processing time: {final_stats['duration']}")
    print(f"   • Model version created: {final_stats['model_version']}")
    print(f"   • Total samples split: {final_stats['total_samples']}")
    print(f"   • Training samples: {final_stats['train_samples']} ({final_stats['train_ratio']:.1%})")
    print(f"   • Validation samples: {final_stats['val_samples']} ({final_stats['val_ratio']:.1%})")
    
    # Data Leakage Prevention Status
    if final_stats['validation_preserved']:
        print(f"   • Validation set: Preserved from previous model (prevents data leakage)")
    else:
        print(f"   • Validation set: Newly created baseline")
    
    # Smart recommendations
    recommendations = get_dataset_splitting_recommendations(final_stats, initial_stats)
    if recommendations:
        print(f"\n💡 Recommendations:")
        for rec in recommendations:
            print(f"   • {rec}")
    
    # Next steps
    next_steps = get_dataset_splitting_next_steps(final_stats)
    if next_steps:
        print(f"\n🎯 Suggested Next Steps:")
        for step in next_steps:
            print(f"   • {step}")
    
    # Files created
    print(f"\n📁 Files Created/Updated:")
    print(f"   • Training dataset (train/images/ and train/labels/)")
    print(f"   • Validation dataset (val/images/ and val/labels/)")
    print(f"   • YOLO configuration (data.yaml)")
    print(f"   • Training history (.training_history.json)")
    
    print("="*60)
    
    # Update project status
    update_dataset_splitting_project_status(final_stats, dataset_dir)

def get_dataset_splitting_recommendations(final_stats, initial_stats):
    """Simple rule-based dataset splitting recommendations"""
    recommendations = []
    
    # Based on dataset size
    if final_stats['total_samples'] < 100:
        recommendations.append("Small dataset - monitor for overfitting during training")
    elif final_stats['total_samples'] > 1000:
        recommendations.append("Large dataset - consider increased batch size and longer training")
    else:
        recommendations.append("Good dataset size for robust model training")
    
    # Based on split ratios
    if final_stats['val_samples'] < 20:
        recommendations.append("Very small validation set - results may be less reliable")
    elif final_stats['val_samples'] > 200:
        recommendations.append("Large validation set - excellent for reliable performance metrics")
    
    # Based on model versioning
    if final_stats['validation_preserved']:
        recommendations.append("Validation set preserved - can compare models reliably")
    else:
        recommendations.append("Baseline validation set established for future comparisons")
    
    return recommendations

def get_dataset_splitting_next_steps(final_stats):
    """Simple workflow recommendations for post-splitting"""
    next_steps = []
    
    if final_stats['total_samples'] >= 100:
        next_steps.append("Run model_training.py with your prepared dataset")
        next_steps.append("Monitor training metrics (loss curves, mAP scores)")
        next_steps.append("Validate model performance on the preserved validation set")
        
        if final_stats['total_samples'] >= 500:
            next_steps.append("Consider advanced training techniques (longer epochs, learning rate scheduling)")
            next_steps.append("Enable early stopping to prevent overfitting")
    else:
        next_steps.append("Consider collecting more training data before final model training")
        next_steps.append("Or proceed with current data for initial baseline model")
        next_steps.append("Monitor closely for overfitting with small dataset")
    
    return next_steps

def update_dataset_splitting_project_status(final_stats, dataset_dir):
    """Update project status file with splitting results"""
    try:
        from datetime import datetime
        dataset_path = Path(dataset_dir)
        status_file = dataset_path / 'PROJECT_STATUS.txt'
        
        splitting_section = f"""
DATASET SPLITTING STATUS:
├── Model Version: {final_stats['model_version']}
├── Total Samples: {final_stats['total_samples']}
├── Training Samples: {final_stats['train_samples']} ({final_stats['train_ratio']:.1%})
├── Validation Samples: {final_stats['val_samples']} ({final_stats['val_ratio']:.1%})
├── Validation Set: {'Preserved' if final_stats['validation_preserved'] else 'New Baseline'}
├── Last Split: {datetime.now().strftime('%Y-%m-%d %H:%M')}
└── Ready for Training: {'YES' if final_stats['total_samples'] >= 50 else 'NO'}

TRAINING PREPARATION:
├── Small Dataset (50+): {'YES' if final_stats['total_samples'] >= 50 else 'NO'}
├── Good Dataset (200+): {'YES' if final_stats['total_samples'] >= 200 else 'NO'}
├── Large Dataset (500+): {'YES' if final_stats['total_samples'] >= 500 else 'NO'}
└── Data Leakage Prevention: {'YES' if final_stats['validation_preserved'] else 'Baseline Set'}
"""
        
        status_content = f"""DOG POOP DETECTION PROJECT STATUS
Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{splitting_section}
NEXT RECOMMENDED ACTIONS:
{get_splitting_status_next_actions(final_stats)}
"""

        with open(status_file, 'w', encoding='utf-8') as f:
            f.write(status_content)
            
        print(f"📄 Updated project status: {status_file}")
        
    except Exception as e:
        print(f"⚠️ Could not update project status file: {e}")
        try:
            simple_content = f"""PROJECT STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M')}
Dataset Splitting Results:
- Model Version: {final_stats['model_version']}
- Total Samples: {final_stats['total_samples']}
- Training: {final_stats['train_samples']} ({final_stats['train_ratio']:.1%})
- Validation: {final_stats['val_samples']} ({final_stats['val_ratio']:.1%})
Ready for Training: {'YES' if final_stats['total_samples'] >= 50 else 'NO'}
"""
            with open(status_file, 'w', encoding='ascii', errors='replace') as f:
                f.write(simple_content)
            print(f"📄 Created simplified project status (ASCII mode)")
        except Exception as e2:
            print(f"⚠️ Status file creation failed completely: {e2}")

def get_splitting_status_next_actions(final_stats):
    """Get next actions for status file"""
    if final_stats['total_samples'] >= 200:
        return "1. Run model_training.py with prepared dataset\n2. Monitor training metrics carefully\n3. Compare performance across model versions"
    elif final_stats['total_samples'] >= 50:
        return "1. Begin initial model training\n2. Monitor for overfitting with smaller dataset\n3. Consider data collection if performance is low"
    else:
        return "1. Collect more training examples\n2. Re-run augmentation for larger dataset\n3. Aim for 100+ samples before serious training"


if __name__ == "__main__":
    # Configuration
    DATASET_DIR = config.get('dataset.name', 'poop_detection_dataset')
    TRAIN_RATIO = config.get('splitting.train_ratio', 0.8)
    VAL_RATIO = config.get('splitting.val_ratio', 0.2)
    RANDOM_SEED = 42
    
    print(get_project_title('splitting'))
    print("=" * 40)
    
    # Check if dataset directory exists
    if not Path(DATASET_DIR).exists():
        print(f"❌ Dataset directory not found: {DATASET_DIR}")
        print("Make sure you've run the annotation tool first!")
        exit(1)
    
    # Store initial state for summary
    from datetime import datetime
    start_time = datetime.now()
    
    # Get initial statistics
    initial_stats = get_initial_dataset_stats(DATASET_DIR)
    
    # Auto-detect and let user choose dataset
    stats = create_yolo_dataset_structure(
        base_dir=DATASET_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=RANDOM_SEED
        # images_subdir and labels_subdir will be auto-detected
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    if stats:
        # Validate the result
        validation_report = validate_dataset_structure(DATASET_DIR)
        
        # Print summary
        print_dataset_summary(stats)
        
        # Enhanced final statistics for session summary
        final_stats = {
            'total_samples': stats['total_samples'],
            'train_samples': stats['train_samples'],
            'val_samples': stats['val_samples'],
            'train_ratio': stats['train_ratio'],
            'val_ratio': stats['val_ratio'],
            'model_version': stats['model_version'],
            'duration': duration,
            'validation_preserved': len(load_training_history(Path(DATASET_DIR))) > 1
        }
        
        # Print comprehensive session summary
        print_dataset_splitting_session_summary(initial_stats, final_stats, DATASET_DIR)
        
        print(f"\n📁 Your dataset is now ready at: {Path(DATASET_DIR).absolute()}")
        print("Next step: Train your YOLO model! 🚀")
    else:
        print("❌ Dataset splitting failed. Check your annotations and try again.")