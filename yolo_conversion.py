from pathlib import Path
import json
from PIL import Image
from collections import Counter

import yaml
from config import config, get_project_title

def convert_labelme_to_yolo(json_dir, output_dir, class_mapping=None):
    """
    Convert labelme JSON annotations to YOLO format with multi-class support
    
    Args:
        json_dir: Directory containing JSON annotation files
        output_dir: Directory to save YOLO format files
        class_mapping: Dictionary mapping class names to IDs (auto-generated if None)
    
    Returns:
        Dictionary with conversion statistics and class mapping
    """
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Auto-detect classes if no mapping provided
    if class_mapping is None:
        print("🔍 Auto-detecting classes from annotations...")
        all_classes = set()
        
        for json_file in Path(json_dir).glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            for shape in data.get('shapes', []):
                if 'label' in shape:
                    all_classes.add(shape['label'])
        
        # Create class mapping (alphabetical order for consistency)
        sorted_classes = sorted(list(all_classes))
        class_mapping = {class_name: idx for idx, class_name in enumerate(sorted_classes)}
        
        print(f"📋 Detected classes: {sorted_classes}")
    
    print(f"🏷️ Class mapping:")
    for class_name, class_id in class_mapping.items():
        print(f"   • {class_name}: {class_id}")
    
    # Statistics tracking
    stats = {
        'total_files': 0,
        'converted_files': 0,
        'total_annotations': 0,
        'class_counts': Counter(),
        'class_mapping': class_mapping,
        'skipped_files': []
    }
    
    # Convert each JSON file
    for json_file in Path(json_dir).glob("*.json"):
        stats['total_files'] += 1
        
        try:
            # Load JSON annotation
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Get image dimensions
            img_path = Path(json_dir).parent / 'images' / data['imagePath']
            if not img_path.exists():
                print(f"⚠️ Image not found for {json_file.name}: {img_path}")
                stats['skipped_files'].append(str(json_file))
                continue
            
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # Create YOLO format file
            yolo_file = Path(output_dir) / f"{json_file.stem}.txt"
            
            annotations_in_file = 0
            
            with open(yolo_file, 'w') as f:
                for shape in data.get('shapes', []):
                    if shape['shape_type'] == 'rectangle' and 'label' in shape:
                        label = shape['label']
                        
                        # Skip unknown classes
                        if label not in class_mapping:
                            print(f"⚠️ Unknown class '{label}' in {json_file.name} - skipping")
                            continue
                        
                        # Convert to YOLO format (normalized coordinates)
                        points = shape['points']
                        x1, y1 = points[0]
                        x2, y2 = points[1]
                        
                        # Calculate center and dimensions (normalized)
                        center_x = (x1 + x2) / 2 / img_width
                        center_y = (y1 + y2) / 2 / img_height
                        width = abs(x2 - x1) / img_width
                        height = abs(y2 - y1) / img_height
                        
                        # Get class ID
                        class_id = class_mapping[label]
                        
                        # Write YOLO format line
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                        
                        annotations_in_file += 1
                        stats['total_annotations'] += 1
                        stats['class_counts'][label] += 1
            
            if annotations_in_file > 0:
                print(f"✅ {json_file.name}: {annotations_in_file} annotations")
                stats['converted_files'] += 1
            else:
                print(f"⚠️ {json_file.name}: No valid annotations found")
                # Remove empty YOLO file
                if yolo_file.exists():
                    yolo_file.unlink()
                    
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")
            stats['skipped_files'].append(str(json_file))
    
    return stats

def create_classes_file(output_dir, class_mapping):
    """
    Create classes.txt file for YOLO training
    
    Args:
        output_dir: Directory to save classes.txt
        class_mapping: Dictionary mapping class names to IDs
    """
    
    classes_file = Path(output_dir).parent / 'classes.txt'
    
    # Sort classes by ID to ensure correct order
    sorted_classes = sorted(class_mapping.items(), key=lambda x: x[1])
    
    with open(classes_file, 'w') as f:
        for class_name, class_id in sorted_classes:
            f.write(f"{class_name}\n")
    
    print(f"📄 Created classes file: {classes_file}")
    return str(classes_file)

def update_data_yaml(dataset_dir, class_mapping):
    """
    Create or update data.yaml with correct configuration
    
    Args:
        dataset_dir: Dataset directory containing data.yaml
        class_mapping: Dictionary mapping class names to IDs
    """
    
    data_yaml_path = Path(dataset_dir) / 'data.yaml'
    
    # Create complete YOLO configuration
    sorted_classes = sorted(class_mapping.items(), key=lambda x: x[1])
    class_names = [class_name for class_name, _ in sorted_classes]
    
    data_config = {
        'path': str(Path(dataset_dir).absolute()),
        'train': 'train/images',
        'val': 'val/images', 
        'nc': len(class_mapping),
        'names': class_names
    }
    
    # Save complete configuration
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"📄 Created complete data.yaml:")
    print(f"   • Classes: {len(class_mapping)}")
    print(f"   • Names: {class_names}")
    print(f"   • Path: {data_config['path']}")
    
    return str(data_yaml_path)

def print_conversion_summary(stats):
    """
    Print a detailed conversion summary
    
    Args:
        stats: Statistics dictionary from convert_labelme_to_yolo
    """
    
    print("\n" + "="*60)
    print("📊 YOLO CONVERSION SUMMARY")
    print("="*60)
    
    print(f"📁 Files:")
    print(f"   • Total JSON files: {stats['total_files']}")
    print(f"   • Successfully converted: {stats['converted_files']}")
    print(f"   • Skipped files: {len(stats['skipped_files'])}")
    
    print(f"\n🏷️ Annotations:")
    print(f"   • Total annotations: {stats['total_annotations']}")
    
    if stats['class_counts']:
        print(f"\n📋 Class Distribution:")
        for class_name, count in stats['class_counts'].most_common():
            class_id = stats['class_mapping'][class_name]
            percentage = (count / stats['total_annotations']) * 100
            print(f"   • {class_name} (ID:{class_id}): {count} ({percentage:.1f}%)")
    
    if stats['skipped_files']:
        print(f"\n⚠️ Skipped Files:")
        for skipped_file in stats['skipped_files']:
            print(f"   • {skipped_file}")
    
    print("="*60)

def validate_image_label_pairs(images_dir, labels_dir):
    """
    Validate which images have corresponding labels
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing JSON labels
        
    Returns:
        Dictionary with validation results
    """
    
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    # Get all images and labels
    image_files = {f.stem for f in images_path.glob("*.jpg")}
    label_files = {f.stem for f in labels_path.glob("*.json")}
    
    # Find matches and mismatches
    matched_pairs = image_files & label_files 
    images_without_labels = image_files - label_files
    labels_without_images = label_files - image_files
    
    print("\n📊 IMAGE-LABEL VALIDATION")
    print("=" * 50)
    print(f"📁 Total images: {len(image_files)}")
    print(f"🏷️ Total labels: {len(label_files)}")
    print(f"✅ Matched pairs: {len(matched_pairs)}")
    print(f"❌ Images without labels: {len(images_without_labels)}")
    print(f"⚠️ Labels without images: {len(labels_without_images)}")
    
    if images_without_labels:
        print(f"\n📝 Unannotated images (will be ignored in training):")
        for filename in sorted(list(images_without_labels)[:10]):
            print(f"   • {filename}.jpg")
        if len(images_without_labels) > 10:
            print(f"   • ... and {len(images_without_labels) - 10} more")
    
    if labels_without_images:
        print(f"\n⚠️ Orphaned labels (missing images):")
        for filename in sorted(labels_without_images):
            print(f"   • {filename}.json")
    
    annotation_rate = len(matched_pairs) / len(image_files) * 100 if image_files else 0
    print(f"\n📈 Annotation Progress: {annotation_rate:.1f}%")
    
    return {
        'total_images': len(image_files),
        'total_labels': len(label_files),
        'matched_pairs': len(matched_pairs),
        'images_without_labels': len(images_without_labels),
        'labels_without_images': len(labels_without_images),
        'annotation_rate': annotation_rate,
        'ready_for_training': len(matched_pairs) > 0
    }

def print_yolo_conversion_session_summary(initial_stats, final_stats, dataset_dir):
    """Generate user-friendly YOLO conversion session summary with recommendations"""
    
    print("\n" + "="*60)
    print("📋 YOLO CONVERSION COMPLETED")
    print("="*60)
    
    # Session Results
    print(f"⏰ Session Results:")
    print(f"   • Processing time: {final_stats['duration']}")
    print(f"   • Files converted: {final_stats['converted_files']}")
    print(f"   • Total annotations: {final_stats['total_annotations']}")
    print(f"   • Classes detected: {final_stats['classes_detected']}")
    print(f"   • Files skipped: {final_stats['skipped_files']}")
    
    # Class Distribution Summary
    if final_stats['class_counts']:
        print(f"\n📊 Class Distribution:")
        for class_name, count in final_stats['class_counts'].most_common():
            percentage = (count / final_stats['total_annotations']) * 100
            print(f"   • {class_name}: {count} annotations ({percentage:.1f}%)")
    
    # Smart recommendations
    recommendations = get_yolo_conversion_recommendations(final_stats, initial_stats)
    if recommendations:
        print(f"\n💡 Recommendations:")
        for rec in recommendations:
            print(f"   • {rec}")
    
    # Next steps
    next_steps = get_yolo_conversion_next_steps(final_stats)
    if next_steps:
        print(f"\n🎯 Suggested Next Steps:")
        for step in next_steps:
            print(f"   • {step}")
    
    # Files created
    print(f"\n📁 Files Created/Updated:")
    print(f"   • YOLO labels directory (yolo_labels/)")
    print(f"   • Classes definition file (classes.txt)")
    print(f"   • YOLO configuration file (data.yaml)")
    
    print("="*60)
    
    # Update project status
    update_yolo_conversion_project_status(final_stats, dataset_dir)

def get_yolo_conversion_recommendations(final_stats, initial_stats):
    """Simple rule-based YOLO conversion recommendations"""
    recommendations = []
    
    # Based on conversion success rate
    if final_stats['skipped_files'] > 0:
        recommendations.append(f"{final_stats['skipped_files']} files were skipped - check for missing images")
    
    # Based on class distribution
    if final_stats['classes_detected'] == 1:
        recommendations.append("Single class detected - consider multi-class annotation for better training")
    elif final_stats['classes_detected'] > 5:
        recommendations.append("Many classes detected - ensure each has sufficient examples")
    
    # Based on annotation count
    if final_stats['total_annotations'] < 100:
        recommendations.append("Low annotation count - consider more examples per class")
    elif final_stats['total_annotations'] > 1000:
        recommendations.append("Large annotation set - excellent for training robust models")
    
    # Class imbalance check
    if final_stats['class_counts']:
        max_count = max(final_stats['class_counts'].values())
        min_count = min(final_stats['class_counts'].values())
        if max_count > min_count * 10:  # 10:1 ratio or more
            recommendations.append("Class imbalance detected - consider augmenting minority classes")
    
    return recommendations

def get_yolo_conversion_next_steps(final_stats):
    """Simple workflow recommendations for post-conversion"""
    next_steps = []
    
    if final_stats['converted_files'] >= 20:
        next_steps.append("Run data_augmentation.py (multiply dataset for better training)")
        next_steps.append("Run dataset_splitting.py (prepare train/validation sets)")
        next_steps.append("Begin model training with converted YOLO format")
        
        if final_stats['total_annotations'] >= 100:
            next_steps.append("Consider advanced training techniques (longer epochs, validation)")
    else:
        next_steps.append("Annotate more images for better model performance")
        next_steps.append("Aim for 50+ annotated images before training")
    
    return next_steps

def update_yolo_conversion_project_status(final_stats, dataset_dir):
    """Update project status file with conversion results"""
    try:
        from datetime import datetime
        dataset_path = Path(dataset_dir)
        status_file = dataset_path / 'PROJECT_STATUS.txt'
        
        # Create conversion status section
        conversion_section = f"""
YOLO CONVERSION STATUS:
├── Files Converted: {final_stats['converted_files']}
├── Total Annotations: {final_stats['total_annotations']}
├── Classes Detected: {final_stats['classes_detected']}
├── Class Names: {list(final_stats['class_mapping'].keys())}
├── Last Converted: {datetime.now().strftime('%Y-%m-%d %H:%M')}
└── Ready for Augmentation: {'YES' if final_stats['converted_files'] >= 10 else 'NO'}

CLASS DISTRIBUTION:
{get_class_distribution_summary(final_stats['class_counts'], final_stats['total_annotations'])}
"""
        
        status_content = f"""DOG POOP DETECTION PROJECT STATUS
Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{conversion_section}
NEXT RECOMMENDED ACTIONS:
{get_conversion_status_next_actions(final_stats)}
"""
        
        # Write with explicit UTF-8 encoding
        with open(status_file, 'w', encoding='utf-8') as f:
            f.write(status_content)
            
        print(f"📄 Updated project status: {status_file}")
        
    except Exception as e:
        print(f"⚠️ Could not update project status file: {e}")
        try:
            simple_content = f"""PROJECT STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M')}
YOLO Conversion Results:
- Files Converted: {final_stats['converted_files']}
- Total Annotations: {final_stats['total_annotations']}
- Classes Detected: {final_stats['classes_detected']}
Ready for Augmentation: {'YES' if final_stats['converted_files'] >= 10 else 'NO'}
"""
            with open(status_file, 'w', encoding='ascii', errors='replace') as f:
                f.write(simple_content)
            print(f"📄 Created simplified project status (ASCII mode)")
        except Exception as e2:
            print(f"⚠️ Status file creation failed completely: {e2}")

def get_class_distribution_summary(class_counts, total_annotations):
    """Generate class distribution summary for status file"""
    if not class_counts:
        return "└── No classes detected"
    
    summary_lines = []
    for class_name, count in class_counts.most_common():
        percentage = (count / total_annotations) * 100
        summary_lines.append(f"├── {class_name}: {count} ({percentage:.1f}%)")
    
    # Change last line to use └──
    if summary_lines:
        summary_lines[-1] = summary_lines[-1].replace('├──', '└──')
    
    return '\n'.join(summary_lines)

def get_conversion_status_next_actions(final_stats):
    """Get next actions for status file"""
    if final_stats['total_annotations'] >= 100:
        return "1. Run data_augmentation.py (recommended 3-5x multiplier)\n2. Run dataset_splitting.py\n3. Begin model training"
    elif final_stats['converted_files'] >= 20:
        return "1. Consider more annotations for better training\n2. Or proceed with data_augmentation.py\n3. Monitor for class imbalance"
    else:
        return "1. Continue annotation work\n2. Aim for 50+ annotated images\n3. Re-run conversion when ready"

if __name__ == "__main__":
    # Configuration
    DATASET_DIR = config.get('dataset.name', 'poop_detection_dataset')
    JSON_LABELS_DIR = f"{DATASET_DIR}/labels"
    IMAGES_DIR = f"{DATASET_DIR}/images"
    YOLO_OUTPUT_DIR = f"{DATASET_DIR}/yolo_labels"
    
    print(get_project_title('training'))
    print("=" * 40)
    
    # Store initial state for summary
    from datetime import datetime
    start_time = datetime.now()
    
    # Validate image-label pairs first
    validation_results = validate_image_label_pairs(IMAGES_DIR, JSON_LABELS_DIR)
    
    initial_stats = {
        'total_images': validation_results['total_images'],
        'total_labels': validation_results['total_labels'],
        'annotation_rate': validation_results['annotation_rate'],
        'ready_for_training': validation_results['ready_for_training']
    }
    
    if not validation_results['ready_for_training']:
        print("❌ No annotated images found! Please annotate some images first.")
        exit(1)
    
    if validation_results['annotation_rate'] < 20:
        proceed = input(f"\n⚠️ Only {validation_results['annotation_rate']:.1f}% of images are annotated. Continue? (y/n): ")
        if proceed.lower() != 'y':
            exit(0)
    
    CUSTOM_CLASSES = None  # Auto-detect classes
    
    # Check if input directory exists
    if not Path(JSON_LABELS_DIR).exists():
        print(f"❌ JSON labels directory not found: {JSON_LABELS_DIR}")
        print("Make sure you've annotated some images first!")
        exit(1)
    
    # Convert annotations
    print(f"🔄 Converting annotations from: {JSON_LABELS_DIR}")
    stats = convert_labelme_to_yolo(
        json_dir=JSON_LABELS_DIR,
        output_dir=YOLO_OUTPUT_DIR,
        class_mapping=CUSTOM_CLASSES
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    if stats['converted_files'] > 0:
        # Create classes.txt file
        classes_file = create_classes_file(YOLO_OUTPUT_DIR, stats['class_mapping'])
        
        # Update data.yaml if it exists
        update_data_yaml(DATASET_DIR, stats['class_mapping'])
        
        # Print conversion summary
        print_conversion_summary(stats)
        
        # Print comprehensive session summary
        final_stats = {
            'converted_files': stats['converted_files'],
            'total_annotations': stats['total_annotations'],
            'classes_detected': len(stats['class_mapping']),
            'class_mapping': stats['class_mapping'],
            'class_counts': stats['class_counts'],
            'duration': duration,
            'skipped_files': len(stats['skipped_files'])
        }
        
        print_yolo_conversion_session_summary(initial_stats, final_stats, DATASET_DIR)
        
    else:
        print("❌ No files were converted. Check your annotations and try again.")