import cv2
import albumentations as A
from pathlib import Path
import shutil
from config import config, get_project_title

class DatasetAugmenter:
    """
    Augment dataset with realistic transformations
    """
    
    def __init__(self, dataset_dir, target_multiplier=5):
        self.dataset_dir = Path(dataset_dir)
        self.target_multiplier = target_multiplier
        
        # Define augmentation pipeline optimized for outdoor scenes
        self.augmentation_pipeline = A.Compose([
            # Lighting variations (very common outdoors)
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.CLAHE(clip_limit=2.0, p=1.0)
            ], p=0.8),
            
            # Weather/atmospheric effects  
            A.OneOf([
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=1.0),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_radius=50, p=1.0)
            ], p=0.4),
            
            # Geometric transforms (mild to preserve object relationships)
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=1.0),
                A.Perspective(scale=(0.05, 0.1), p=1.0),
                A.ElasticTransform(alpha=50, sigma=5, p=1.0)
            ], p=0.6),
            
            # Image quality variations (camera effects)
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.Blur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0)
            ], p=0.5),
            
            # Color variations (different times of day, seasons)
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                A.ChannelShuffle(p=1.0)
            ], p=0.4),
            
            # Occasional flips (horizontal only - vertical would be unnatural)
            A.HorizontalFlip(p=0.3)
            
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def augment_dataset(self):
        """
        Create augmented versions of the dataset
        """
        
        print(f"🔄 Augmenting Dataset")
        print("=" * 50)
        
        original_images = self.dataset_dir / 'images'
        original_labels = self.dataset_dir / 'yolo_labels'
        
        # Create augmented directories
        aug_images_dir = self.dataset_dir / 'augmented_images'
        aug_labels_dir = self.dataset_dir / 'augmented_labels'
        
        aug_images_dir.mkdir(exist_ok=True)
        aug_labels_dir.mkdir(exist_ok=True)
        
        # First, copy original images
        print("📁 Copying original dataset...")
        self.copy_original_files(original_images, original_labels, aug_images_dir, aug_labels_dir)
        
        # Get list of image files that have corresponding labels
        print("🔍 Finding annotated images...")
        all_image_files = list(original_images.glob('*.jpg'))
        image_files = []

        for img_file in all_image_files:
            label_file = original_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                image_files.append(img_file)

        print(f"📊 Found {len(image_files)} annotated images out of {len(all_image_files)} total images")

        if len(image_files) != len(all_image_files):
            skipped = len(all_image_files) - len(image_files)
            print(f"⏭️ Skipping {skipped} images without annotations")

        total_augmentations = len(image_files) * (self.target_multiplier - 1)
        
        print(f"🎯 Creating {total_augmentations} augmented images...")
        print(f"   • Original: {len(image_files)} images")
        print(f"   • Target: {len(image_files) * self.target_multiplier} total images")
        
        augmented_count = 0
        
        for img_file in image_files:
            label_file = original_labels / f"{img_file.stem}.txt"
            
            if not label_file.exists():
                print(f"⚠️ Skipping {img_file.name} - no label file")
                continue
            
            # Load image
            image = cv2.imread(str(img_file))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load bounding boxes
            bboxes, class_labels = self.load_yolo_annotations(label_file)
            
            if not bboxes:
                print(f"⚠️ Skipping {img_file.name} - no annotations")
                continue
            
            # Create augmented versions
            for aug_idx in range(self.target_multiplier - 1):
                try:
                    # Apply augmentation
                    augmented = self.augmentation_pipeline(
                        image=image_rgb,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_class_labels = augmented['class_labels']
                    
                    # Skip if augmentation removed all bboxes
                    if not aug_bboxes:
                        continue
                    
                    # Save augmented image
                    aug_img_name = f"{img_file.stem}_aug_{aug_idx:02d}.jpg"
                    aug_img_path = aug_images_dir / aug_img_name
                    
                    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(aug_img_path), aug_image_bgr)
                    
                    # Save augmented labels
                    aug_label_path = aug_labels_dir / f"{img_file.stem}_aug_{aug_idx:02d}.txt"
                    self.save_yolo_annotations(aug_label_path, aug_bboxes, aug_class_labels)
                    
                    augmented_count += 1
                    
                    if augmented_count % 20 == 0:
                        print(f"   ✅ Generated {augmented_count}/{total_augmentations} augmented images")
                
                except Exception as e:
                    print(f"   ⚠️ Failed to augment {img_file.name} (attempt {aug_idx}): {e}")
                    continue
        
        print(f"\n📊 Augmentation Summary:")
        total_images = len(list(aug_images_dir.glob('*.jpg')))
        total_labels = len(list(aug_labels_dir.glob('*.txt')))
        print(f"   • Total images: {total_images}")
        print(f"   • Total labels: {total_labels}")
        print(f"   • Augmentation ratio: {total_images / len(image_files):.1f}x")
        
        return aug_images_dir, aug_labels_dir 
    
    def copy_original_files(self, orig_img_dir, orig_lbl_dir, aug_img_dir, aug_lbl_dir):
        """Copy original files to augmented directories"""
        
        for img_file in orig_img_dir.glob('*.jpg'):
            shutil.copy2(img_file, aug_img_dir / img_file.name)
        
        for lbl_file in orig_lbl_dir.glob('*.txt'):
            shutil.copy2(lbl_file, aug_lbl_dir / lbl_file.name)
    
    def load_yolo_annotations(self, label_file):
        """Load YOLO format annotations"""
        
        bboxes = []
        class_labels = []
        
        with open(label_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
        
        return bboxes, class_labels
    
    def save_yolo_annotations(self, label_file, bboxes, class_labels):
        """Save YOLO format annotations"""
        
        with open(label_file, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                x_center, y_center, width, height = bbox
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def print_augmentation_session_summary(initial_stats, final_stats, dataset_dir):
    """Generate user-friendly augmentation session summary with recommendations"""
    
    print("\n" + "="*60)
    print("📋 DATA AUGMENTATION COMPLETED")
    print("="*60)
    
    # Session Results
    print(f"⏰ Session Results:")
    print(f"   • Processing time: {final_stats['duration']}")
    print(f"   • Augmentation multiplier: {final_stats['multiplier']}x")
    print(f"   • Images before: {initial_stats['annotated_images']}")
    print(f"   • Images after: {final_stats['total_images']}")
    print(f"   • Actual multiplier achieved: {final_stats['augmentation_ratio']:.1f}x")
    
    # Quality Check
    image_label_match = final_stats['total_images'] == final_stats['total_labels']
    print(f"   • Image-label pairs match: {'✅' if image_label_match else '❌'}")
    
    # Smart recommendations
    recommendations = get_augmentation_recommendations(final_stats, initial_stats)
    if recommendations:
        print(f"\n💡 Recommendations:")
        for rec in recommendations:
            print(f"   • {rec}")
    
    # Next steps
    next_steps = get_augmentation_next_steps(final_stats)
    if next_steps:
        print(f"\n🎯 Suggested Next Steps:")
        for step in next_steps:
            print(f"   • {step}")
    
    # Files created
    print(f"\n📁 Files Updated:")
    print(f"   • Created augmented_images/ ({final_stats['total_images']} images)")
    print(f"   • Created augmented_labels/ ({final_stats['total_labels']} labels)")
    
    print("="*60)
    
    # Update project status
    update_augmentation_project_status(final_stats, dataset_dir)

def get_augmentation_recommendations(final_stats, initial_stats):
    """Simple rule-based augmentation recommendations"""
    recommendations = []
    
    # Based on final dataset size
    if final_stats['total_images'] < 200:
        recommendations.append("Consider higher multiplier next time for better model performance")
    elif final_stats['total_images'] > 2000:
        recommendations.append("Large dataset! Consider reducing epochs or increasing batch size")
    else:
        recommendations.append("Good dataset size for training robust models")
    
    # Based on augmentation ratio
    if final_stats['augmentation_ratio'] < final_stats['multiplier'] * 0.9:
        recommendations.append("Some augmentations may have failed - check logs above")
    
    # Based on image-label consistency
    if final_stats['total_images'] != final_stats['total_labels']:
        recommendations.append("Image-label mismatch detected - verify augmentation quality")
    
    return recommendations

def get_augmentation_next_steps(final_stats):
    """Simple workflow recommendations for post-augmentation"""
    next_steps = []
    
    if final_stats['total_images'] >= 100:
        next_steps.append("Run dataset_splitting.py (prepare augmented train/validation sets)")
        next_steps.append("Update model training to use augmented dataset")
        next_steps.append("Consider longer training (more epochs) with larger dataset")
        
        if final_stats['total_images'] >= 500:
            next_steps.append("Enable early stopping to prevent overfitting")
            next_steps.append("Consider increasing batch size for faster training")
    else:
        next_steps.append("Dataset still small - consider more augmentation or more base images")
        next_steps.append("Collect more diverse examples before training")
    
    return next_steps

def update_augmentation_project_status(final_stats, dataset_dir):
    """Update project status file with augmentation results"""
    try:
        from datetime import datetime
        dataset_path = Path(dataset_dir)
        status_file = dataset_path / 'PROJECT_STATUS.txt'
        
        augmentation_section = f"""
AUGMENTATION STATUS:
├── Dataset Multiplier: {final_stats['multiplier']}x (target)
├── Actual Multiplier: {final_stats['augmentation_ratio']:.1f}x (achieved)
├── Total Images: {final_stats['total_images']}
├── Total Labels: {final_stats['total_labels']}
├── Last Augmented: {datetime.now().strftime('%Y-%m-%d %H:%M')}
└── Ready for Training: {'YES' if final_stats['total_images'] >= 100 else 'NO'}

TRAINING READINESS:
├── Minimum Dataset: {'YES' if final_stats['total_images'] >= 50 else 'NO (Need 50+)'}
├── Good Dataset: {'YES' if final_stats['total_images'] >= 200 else 'NO (Need 200+)'}
└── Large Dataset: {'YES' if final_stats['total_images'] >= 500 else 'NO (Need 500+)'}
"""
        
        status_content = f"""DOG POOP DETECTION PROJECT STATUS
Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{augmentation_section}
NEXT RECOMMENDED ACTIONS:
{get_status_next_actions(final_stats)}
"""
        
        # Write with explicit UTF-8 encoding
        with open(status_file, 'w', encoding='utf-8') as f:
            f.write(status_content)
            
        print(f"📄 Updated project status: {status_file}")
        
    except Exception as e:
        print(f"⚠️ Could not update project status file: {e}")
        try:
            simple_content = f"""PROJECT STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M')}
Augmentation Results:
- Dataset Multiplier: {final_stats['multiplier']}x
- Total Images: {final_stats['total_images']}
- Total Labels: {final_stats['total_labels']}
Ready for Training: {'YES' if final_stats['total_images'] >= 100 else 'NO'}
"""
            with open(status_file, 'w', encoding='ascii', errors='replace') as f:
                f.write(simple_content)
            print(f"📄 Created simplified project status (ASCII mode)")
        except Exception as e2:
            print(f"⚠️ Status file creation failed completely: {e2}")

def get_status_next_actions(final_stats):
    """Get next actions for status file"""
    if final_stats['total_images'] >= 200:
        return "1. Run dataset_splitting.py with augmented data\n2. Train model with longer epochs\n3. Monitor for overfitting"
    elif final_stats['total_images'] >= 50:
        return "1. Run dataset_splitting.py\n2. Train initial model\n3. Evaluate performance"
    else:
        return "1. Collect more base images\n2. Re-run augmentation with higher multiplier\n3. Aim for 100+ total images"

if __name__ == "__main__":
    print(get_project_title('augmentation'))
    print("=" * 40)
    
    # Configuration
    DATASET_DIR = config.get('dataset.name', 'poop_detection_dataset')
    
    # Get current dataset size (annotated images only)
    images_dir = Path(DATASET_DIR) / 'images'
    labels_dir = Path(DATASET_DIR) / 'yolo_labels'

    # Count only images that have corresponding labels
    all_images = list(images_dir.glob('*.jpg'))
    annotated_images = []
    for img_file in all_images:
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            annotated_images.append(img_file)

    current_count = len(annotated_images)
    total_images = len(all_images)

    print(f"📊 Current dataset: {current_count} annotated images (out of {total_images} total)")
    
    # Store initial state for summary
    initial_stats = {
        'annotated_images': current_count,
        'total_images': total_images,
        'annotation_rate': (current_count / total_images * 100) if total_images > 0 else 0
    }
    
    # Ask user for augmentation factor
    print(f"💡 Recommended multiplier for small datasets: 3-5x")
    multiplier = input(f"Enter augmentation multiplier (default: 4): ").strip()
    multiplier = int(multiplier) if multiplier.isdigit() else 4
    
    target_size = current_count * multiplier
    print(f"🎯 Target dataset size: {target_size} images")
    
    proceed = input("Proceed with augmentation? (y/n): ").lower().strip()
    if proceed != 'y':
        print("Augmentation cancelled")
        exit()
    
    # Create augmenter and run (with timing)
    from datetime import datetime
    start_time = datetime.now()
    
    augmenter = DatasetAugmenter(DATASET_DIR, target_multiplier=multiplier)
    aug_images, aug_labels = augmenter.augment_dataset()
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Get final statistics
    final_images = len(list(aug_images.glob('*.jpg')))
    final_labels = len(list(aug_labels.glob('*.txt')))
    
    final_stats = {
        'total_images': final_images,
        'total_labels': final_labels,
        'augmentation_ratio': final_images / current_count if current_count > 0 else 0,
        'duration': duration,
        'multiplier': multiplier
    }
    
    # Print comprehensive session summary
    print_augmentation_session_summary(initial_stats, final_stats, DATASET_DIR)