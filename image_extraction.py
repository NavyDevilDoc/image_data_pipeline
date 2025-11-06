import cv2
from config import get_project_title
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import json
import shutil
from config import config


# NEW: Status tracking functions
def create_status_files(dataset_dir):
    """
    Create status tracking files if they don't exist
    
    Args:
        dataset_dir: Path to dataset directory
    """
    dataset_path = Path(dataset_dir)
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Create annotation status file
    status_file = dataset_path / '.annotation_status.json'
    if not status_file.exists():
        with open(status_file, 'w') as f:
            json.dump({}, f, indent=2)
    
    # Create extraction log file
    log_file = dataset_path / '.extraction_log.json'
    if not log_file.exists():
        with open(log_file, 'w') as f:
            json.dump({}, f, indent=2)

def scan_media_in_new():
    """
    Scan for videos and photos in raw_media/New/ directory
    
    Returns:
        Tuple of (video_files, photo_files)
    """
    new_media_dir = Path("raw_media/New")
    
    if not new_media_dir.exists():
        print(f"📁 Creating directory: {new_media_dir}")
        new_media_dir.mkdir(parents=True, exist_ok=True)
        return [], []
    
    # Find video files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(new_media_dir.glob(ext))
    
    # Find photo files
    photo_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    photo_files = []
    for ext in photo_extensions:
        photo_files.extend(new_media_dir.glob(ext))
    
    return [str(f) for f in video_files], [str(f) for f in photo_files]

def move_media_to_processed(media_path):
    """
    Move video or photo from New/ to Processed/ after successful processing
    
    Args:
        media_path: Path to media file in New/
        
    Returns:
        Path to moved media file
    """
    media_path = Path(media_path)
    processed_dir = Path("raw_media/Processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    destination = processed_dir / media_path.name
    
    # Handle naming conflicts
    counter = 1
    while destination.exists():
        stem = media_path.stem
        suffix = media_path.suffix
        destination = processed_dir / f"{stem}_{counter:03d}{suffix}"
        counter += 1
    
    try:
        shutil.move(str(media_path), str(destination))
        print(f"   📦 Moved to: {destination}")
        return str(destination)
    except Exception as e:
        print(f"   ❌ Failed to move media: {e}")
        return str(media_path)

def log_extraction_session(dataset_dir, video_files, frames_extracted, extraction_dirs):
    """
    Log extraction session details
    
    Args:
        dataset_dir: Dataset directory path
        video_files: List of processed video files
        frames_extracted: Total frames extracted
        extraction_dirs: List of temporary extraction directories
    """
    log_file = Path(dataset_dir) / '.extraction_log.json'
    
    # Load existing log
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = {}
    
    # Create session entry
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    log_data[session_id] = {
        'date': datetime.now().isoformat(),
        'videos_processed': [Path(v).name for v in video_files],
        'frames_extracted': frames_extracted,
        'extraction_dirs': extraction_dirs,
        'target_frames_per_video': config.get('extraction.target_frames_per_video', 50)
    }
    
    # Save updated log
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    return session_id

def update_image_status(dataset_dir, new_image_files, session_id):
    """
    Mark newly extracted images as unannotated
    
    Args:
        dataset_dir: Dataset directory path
        new_image_files: List of new image file names
        session_id: Extraction session ID
    """
    status_file = Path(dataset_dir) / '.annotation_status.json'
    
    # Load existing status
    if status_file.exists():
        with open(status_file, 'r') as f:
            status_data = json.load(f)
    else:
        status_data = {}
    
    # Add new images as unannotated
    for image_file in new_image_files:
        image_name = Path(image_file).name
        status_data[image_name] = {
            'status': 'unannotated',
            'extraction_session': session_id,
            'extraction_date': datetime.now().isoformat()
        }
    
    # Save updated status
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2)


def extract_frames_from_video(video_path, output_dir, target_frames=50):
    """
    Extract frames from video using ratio method (evenly distributed)
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        target_frames: Number of frames to extract (default: 50)
    
    Returns:
        Number of frames extracted
    """
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return 0
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"   📹 Duration: {duration:.1f}s, FPS: {fps:.1f}, Frames: {total_frames}")
    
    # Calculate frame extraction ratio
    if total_frames <= target_frames:
        frame_interval = 1
        actual_frames = total_frames
    else:
        frame_interval = total_frames // target_frames
        actual_frames = target_frames
    
    print(f"   🎯 Extracting every {frame_interval} frame(s) for ~{actual_frames} total frames")
    
    # Extract frames
    frame_count = 0
    extracted_count = 0
    video_name = Path(video_path).stem
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at intervals
        if frame_count % frame_interval == 0:
            frame_filename = f"{video_name}_frame_{extracted_count:04d}.jpg"
            frame_path = output_dir / frame_filename
            
            cv2.imwrite(str(frame_path), frame)
            extracted_count += 1
            
            if extracted_count >= target_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"   ✅ Extracted {extracted_count} frames")
    return extracted_count

def process_photos(photo_paths, output_dir):
    """
    Copy and optionally resize photos for annotation
    
    Args:
        photo_paths: List of photo file paths
        output_dir: Directory to save processed photos
    
    Returns:
        Number of photos processed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    for photo_path in photo_paths:
        photo_path = Path(photo_path)
        
        # Read image
        img = cv2.imread(str(photo_path))
        if img is None:
            print(f"   ⚠️ Could not read: {photo_path.name}")
            continue
        
        # Get image properties
        height, width = img.shape[:2]
        file_size = photo_path.stat().st_size / (1024*1024)  # MB
        
        print(f"   📷 {photo_path.name}: {width}×{height}, {file_size:.1f}MB")
        
        # Optional: Resize if image is very large (>4K)
        if width > 4000 or height > 3000:
            scale_factor = min(4000/width, 3000/height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"      🔄 Resized to: {new_width}×{new_height}")
        
        # Save processed photo
        output_filename = f"photo_{photo_path.stem}.jpg"
        output_path = output_dir / output_filename
        
        cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        processed_count += 1
    
    print(f"   ✅ Processed {processed_count} photos")
    return processed_count

def cleanup_extraction_directories(extraction_dirs, force=False):
    """
    Clean up temporary frame extraction directories
    
    Args:
        extraction_dirs: List of extraction directory paths
        force: If True, delete without asking
    
    Returns:
        Number of directories cleaned up
    """
    
    if not extraction_dirs:
        return 0
    
    print(f"\n🗑️ Cleanup Options:")
    print(f"Found {len(extraction_dirs)} temporary extraction directories:")
    
    for i, dir_name in enumerate(extraction_dirs, 1):
        dir_path = Path(dir_name)
        if dir_path.exists():
            frame_count = len(list(dir_path.glob("*.jpg")))
            dir_size = sum(f.stat().st_size for f in dir_path.glob("*.jpg")) / (1024*1024)  # MB
            print(f"   {i}. {dir_name} ({frame_count} frames, {dir_size:.1f} MB)")
    
    if not force:
        cleanup_choice = input("\nCleanup options:\n"
                              "1. Delete all extraction directories\n"
                              "2. Keep all directories\n"
                              "3. Select specific directories to delete\n"
                              "Choice (1-3): ").strip()
    else:
        cleanup_choice = "1"
    
    cleaned_count = 0
    
    if cleanup_choice == "1":
        # Delete all
        print("🗑️ Cleaning up all extraction directories...")
        for dir_name in extraction_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                try:
                    import shutil
                    shutil.rmtree(dir_path)
                    print(f"   ✅ Deleted: {dir_name}")
                    cleaned_count += 1
                except Exception as e:
                    print(f"   ❌ Failed to delete {dir_name}: {e}")
    
    elif cleanup_choice == "2":
        # Keep all
        print("📁 Keeping all extraction directories")
        
    elif cleanup_choice == "3":
        # Selective deletion
        indices_input = input("Enter directory numbers to delete (comma-separated, e.g., 1,3): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in indices_input.split(',')]
            
            for idx in indices:
                if 0 <= idx < len(extraction_dirs):
                    dir_name = extraction_dirs[idx]
                    dir_path = Path(dir_name)
                    if dir_path.exists():
                        try:
                            import shutil
                            shutil.rmtree(dir_path)
                            print(f"   ✅ Deleted: {dir_name}")
                            cleaned_count += 1
                        except Exception as e:
                            print(f"   ❌ Failed to delete {dir_name}: {e}")
                else:
                    print(f"   ⚠️ Invalid index: {idx + 1}")
        except ValueError:
            print("❌ Invalid input format")
    
    else:
        print("❌ Invalid choice, keeping all directories")
    
    if cleaned_count > 0:
        print(f"✅ Cleaned up {cleaned_count} directories")
    
    return cleaned_count

def select_media():
    """
    Auto-scan for videos and photos in New/ directory, with fallback to manual selection
    """
    # First, try to auto-scan New/ directory
    video_files, photo_files = scan_media_in_new()
    
    total_media = len(video_files) + len(photo_files)
    
    if total_media > 0:
        print(f"\n📁 Found media in raw_media/New/:")
        
        if video_files:
            print(f"   📹 Videos ({len(video_files)}):")
            for i, video in enumerate(video_files, 1):
                video_name = Path(video).name
                file_size = Path(video).stat().st_size / (1024*1024)  # MB
                print(f"      {i}. {video_name} ({file_size:.1f} MB)")
        
        if photo_files:
            print(f"   📷 Photos ({len(photo_files)}):")
            for i, photo in enumerate(photo_files, 1):
                photo_name = Path(photo).name
                file_size = Path(photo).stat().st_size / (1024*1024)  # MB
                print(f"      {i}. {photo_name} ({file_size:.1f} MB)")
        
        process_all = input(f"\nProcess all {total_media} media files? (y/n): ").lower().strip()
        
        if process_all == 'y':
            return video_files, photo_files
        else:
            print("⏭️ Skipping auto-processing")
    else:
        print("📁 No media found in raw_media/New/")
    
    # Fallback to manual selection
    print("🔍 Manual media selection...")
    root = tk.Tk()
    root.withdraw()
    
    choice = input("Select (1) videos, (2) photos, or (3) mixed media: ").strip()
    
    if choice == "1":
        file_paths = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv")]
        )
        return list(file_paths), []
    elif choice == "2":
        file_paths = filedialog.askopenfilenames(
            title="Select Photo Files", 
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        return [], list(file_paths)
    else:
        file_paths = filedialog.askopenfilenames(
            title="Select Media Files",
            filetypes=[
                ("All media", "*.mp4 *.avi *.mov *.mkv *.wmv *.jpg *.jpeg *.png *.bmp *.tiff"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")
            ]
        )
        
        # Separate videos and photos
        videos = []
        photos = []
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
        
        for file_path in file_paths:
            if Path(file_path).suffix.lower() in video_exts:
                videos.append(file_path)
            else:
                photos.append(file_path)
        
        return videos, photos
    
    root.destroy()
    return [], []

def preview_frames(frames_dir):
    """
    Show sample of extracted frames
    """
    frame_files = list(Path(frames_dir).glob("*.jpg"))
    if not frame_files:
        return
    
    # Show 6 sample frames
    indices = np.linspace(0, len(frame_files)-1, 6, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        img = cv2.imread(str(frame_files[idx]))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(img_rgb)
        axes[i].set_title(frame_files[idx].name, fontsize=8)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def setup_annotation_directories(extracted_frames_dir, dataset_name="poop_detection_dataset"):
    """
    Copy extracted frames to annotation-ready directory structure
    
    Args:
        extracted_frames_dir: Directory with extracted frames
        dataset_name: Name for the annotation dataset
        
    Returns:
        Path to images directory ready for annotation
    """
    
    # Create dataset structure
    dataset_path = Path(dataset_name)
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy frames to images directory
    extracted_path = Path(extracted_frames_dir)
    frame_files = list(extracted_path.glob("*.jpg"))
    
    print(f"\n📁 Setting up annotation directories...")
    print(f"   • Dataset: {dataset_path}")
    print(f"   • Images: {images_dir}")
    print(f"   • Labels: {labels_dir}")
    
    copied_count = 0
    for frame_file in frame_files:
        dst_path = images_dir / frame_file.name
        if not dst_path.exists():  # Avoid duplicates
            import shutil
            shutil.copy2(frame_file, dst_path)
            copied_count += 1
    
    print(f"   • Copied {copied_count} frames for annotation")
    
    return str(images_dir), str(labels_dir)

# Main execution
if __name__ == "__main__":
    TARGET_FRAMES = config.get('extraction.target_frames_per_video', 50)
    DATASET_NAME = config.get('dataset.name', 'poop_detection_dataset')
    
    print(get_project_title('training'))
    print("=" * 45)
    
    # Create status tracking files
    create_status_files(DATASET_NAME)
    
    # Select media (auto-scan New/ or manual selection)
    video_files, photo_files = select_media()
    total_media = len(video_files) + len(photo_files)
    
    if total_media == 0:
        print("❌ No media selected.")
        exit()
    
    print(f"\n🎬 Processing {len(video_files)} video(s) and {len(photo_files)} photo(s)...")
    
    # Process each video and/or photo
    total_extracted = 0
    all_extracted_dirs = []
    processed_media = []
    
    # Process videos
    for i, video_path in enumerate(video_files, 1):
        print(f"\n📹 Video {i}/{len(video_files)}:")
        print("-" * 30)
        
        # Create timestamped output directory for this video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        output_dir = f"extracted_frames_{video_name}_{timestamp}"
        
        num_extracted = extract_frames_from_video(
            video_path=video_path,
            output_dir=output_dir,
            target_frames=TARGET_FRAMES
        )
        
        if num_extracted > 0:
            total_extracted += num_extracted
            all_extracted_dirs.append(output_dir)
            
            # Move video to Processed/ after successful extraction
            if Path(video_path).parent.name == "New":
                moved_path = move_media_to_processed(video_path)
                processed_media.append(moved_path)
            else:
                processed_media.append(video_path)
    
    # Process photos
    if photo_files:
        print(f"\n📷 Processing {len(photo_files)} photos:")
        print("-" * 30)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        photo_output_dir = f"processed_photos_{timestamp}"
        
        num_processed = process_photos(photo_files, photo_output_dir)
        
        if num_processed > 0:
            total_extracted += num_processed
            all_extracted_dirs.append(photo_output_dir)
            
            # Move photos to Processed/
            for photo_path in photo_files:
                if Path(photo_path).parent.name == "New":
                    moved_path = move_media_to_processed(photo_path)
                    processed_media.append(moved_path)
                else:
                    processed_media.append(photo_path)
    
    # Show preview for first extraction directory
    if all_extracted_dirs:
        print("\n🖼️ Sample images:")
        preview_frames(all_extracted_dirs[0])
    
    print(f"\n🏁 Complete! Total frames extracted: {total_extracted}")
    
    # Ask if user wants to set up for annotation
    if total_extracted > 0:
        setup_annotation = input(f"\n📝 Set up annotation directories? (y/n): ").lower().strip()
        
        if setup_annotation == 'y':
            new_image_files = []
            
            # If multiple extraction dirs, let user choose or merge all
            if len(all_extracted_dirs) == 1:
                images_dir, labels_dir = setup_annotation_directories(all_extracted_dirs[0], DATASET_NAME)
                new_image_files = list(Path(all_extracted_dirs[0]).glob("*.jpg"))
            else:
                print(f"\n📁 Multiple extraction directories:")
                for i, dir_name in enumerate(all_extracted_dirs, 1):
                    frame_count = len(list(Path(dir_name).glob("*.jpg")))
                    print(f"   {i}. {dir_name} ({frame_count} frames)")
                
                choice = input("Choose (1) merge all, (2) select specific, or (3) skip: ").strip()
                
                if choice == "1":
                    # Merge all directories
                    temp_merge_dir = "temp_merged_frames"
                    Path(temp_merge_dir).mkdir(exist_ok=True)
                    
                    for extract_dir in all_extracted_dirs:
                        for frame_file in Path(extract_dir).glob("*.jpg"):
                            shutil.copy2(frame_file, Path(temp_merge_dir) / frame_file.name)
                            new_image_files.append(frame_file)
                    
                    images_dir, labels_dir = setup_annotation_directories(temp_merge_dir, DATASET_NAME)
                    
                    # Clean up temp directory
                    shutil.rmtree(temp_merge_dir)
                    
                elif choice == "2":
                    try:
                        idx = int(input("Enter directory number: ")) - 1
                        images_dir, labels_dir = setup_annotation_directories(all_extracted_dirs[idx], DATASET_NAME)
                        new_image_files = list(Path(all_extracted_dirs[idx]).glob("*.jpg"))
                    except:
                        print("❌ Invalid selection")
                        exit()
                else:
                    print("⏭️ Skipping annotation setup")
                    exit()
            
            # Log extraction session and update image status
            session_id = log_extraction_session(DATASET_NAME, processed_media, total_extracted, all_extracted_dirs)
            update_image_status(DATASET_NAME, new_image_files, session_id)
            
            # Cleanup extraction directories
            cleanup_extraction_directories(all_extracted_dirs)
            
            print(f"\n✅ Ready for annotation!")
            print(f"📁 Session logged: {session_id}")
            print(f"📁 Next steps:")
            print(f"   1. Run: python annotation_tool.py")
            print(f"   2. Annotate images in: {images_dir}")
            print(f"   3. Labels will be saved to: {labels_dir}")
        else:
            print("⏭️ Annotation setup skipped")
            
            # Still offer cleanup even if annotation setup was skipped
            cleanup_extraction_directories(all_extracted_dirs)

    print("📁 Ready for next step!")