import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import json
from pathlib import Path
import shutil
from datetime import datetime
from config import config, get_project_title

class MultiClassAnnotator:
    def __init__(self, images_dir, labels_dir, classes=config.get('dataset.classes', ['poop', 'leaf', 'stick', 'rock']), mode='new_only'):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.mode = mode
        
        # Load annotation status
        self.annotation_status = self.load_annotation_status()
        
        # Filter images based on mode
        all_images = list(self.images_dir.glob("*.jpg"))
        self.image_files = self.filter_images_by_status(all_images, mode)
        
        self.current_idx = 0
        self.annotations = {}
        self.bbox_coords = []
        self.current_boxes = []
        self.is_dragging = False
        self.drag_start = None
        self.preview_rect = None
        self.zoom_level = 1.0
        self.original_xlim = None
        self.original_ylim = None
        self.is_panning = False
        self.pan_start = None
        
        # Class management
        self.classes = classes
        self.current_class = classes[0]  # Default to first class
        # Generate colors dynamically for any class list
        self.class_colors = config.generate_dynamic_colors(self.classes)
        
        # Create labels directory
        self.labels_dir.mkdir(exist_ok=True)
        
        # Show filtered results
        total_images = len(all_images)
        filtered_count = len(self.image_files)
        print(f"📁 Total images: {total_images}")
        print(f"📋 Selected for annotation: {filtered_count} ({mode} mode)")
        print(f"🏷️ Classes: {', '.join(self.classes)}")

    def load_annotation_status(self):
        """Load annotation status from JSON file"""
        dataset_dir = self.images_dir.parent
        status_file = dataset_dir / '.annotation_status.json'
        
        if status_file.exists():
            with open(status_file, 'r') as f:
                return json.load(f)
        else:
            return {}

    def filter_images_by_status(self, all_images, mode):
        """Filter images based on annotation mode"""
        if mode == 'all':
            return all_images
        
        elif mode == 'new_only':
            # Show only unannotated images
            filtered = []
            for img_path in all_images:
                img_name = img_path.name
                status = self.annotation_status.get(img_name, {})
                if status.get('status', 'unannotated') == 'unannotated':
                    filtered.append(img_path)
            return filtered
        
        elif mode == 'resume':
            return self.filter_images_by_status(all_images, 'new_only')
        
        else:
            return all_images

    def update_annotation_status(self, image_name):
        """Mark image as annotated in status file"""
        dataset_dir = self.images_dir.parent
        status_file = dataset_dir / '.annotation_status.json'
        
        # Update status in memory
        if image_name not in self.annotation_status:
            self.annotation_status[image_name] = {}
        
        self.annotation_status[image_name]['status'] = 'annotated'
        self.annotation_status[image_name]['annotated_date'] = datetime.now().isoformat()
        
        # Save to file
        with open(status_file, 'w') as f:
            json.dump(self.annotation_status, f, indent=2)


    def update_annotation_status_archived(self, image_name):
        """Mark image as archived in status file"""
        dataset_dir = self.images_dir.parent
        status_file = dataset_dir / '.annotation_status.json'
        
        # Update status in memory
        if image_name not in self.annotation_status:
            self.annotation_status[image_name] = {}
        
        self.annotation_status[image_name]['status'] = 'archived'
        self.annotation_status[image_name]['archived_date'] = datetime.now().isoformat()
        
        # Save to file
        with open(status_file, 'w') as f:
            json.dump(self.annotation_status, f, indent=2)


    def get_dataset_statistics(self):
        """Get current dataset annotation statistics"""
        all_images = list(self.images_dir.glob("*.jpg"))
        
        # Also check archived images
        archive_dir = self.images_dir.parent / 'archived_images'
        archived_images = list(archive_dir.glob("*.jpg")) if archive_dir.exists() else []
        
        total = len(all_images) + len(archived_images)
        
        annotated_count = 0
        archived_count = 0
        
        # Count current images
        for img_path in all_images:
            img_name = img_path.name
            status = self.annotation_status.get(img_name, {})
            if status.get('status', 'unannotated') == 'annotated':
                annotated_count += 1
        
        # Count archived images
        for img_path in archived_images:
            img_name = img_path.name
            status = self.annotation_status.get(img_name, {})
            if status.get('status', 'unannotated') == 'archived':
                archived_count += 1
        
        unannotated_count = len(all_images) - annotated_count
        
        return {
            'total': total,
            'annotated': annotated_count,
            'archived': archived_count,
            'unannotated': unannotated_count,
            'percentage': (annotated_count / len(all_images) * 100) if len(all_images) > 0 else 0
        }
        
    def start_annotation(self):
        """Start the annotation process"""
        if not self.image_files:
            print("❌ No images found!")
            return
            
        self.show_image()
        
    def show_image(self):
        """Display current image for annotation"""
        if self.current_idx >= len(self.image_files):
            print("🎉 All images in current mode completed!")
            
            # Show final statistics
            stats = self.get_dataset_statistics()
            print(f"\n📊 Final Statistics:")
            print(f"   • Total images: {stats['total']}")
            print(f"   • Annotated: {stats['annotated']} ({stats['percentage']:.1f}%)")
            print(f"   • Remaining: {stats['unannotated']}")
            
            plt.close('all')
            return
            
        # Close previous figure if exists
        plt.close('all')
        
        # Load and display image
        img_path = self.image_files[self.current_idx]
        img = plt.imread(img_path)
        
        # Create figure with more space for class buttons
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        plt.subplots_adjust(bottom=0.25) 
        
    # Initialize pan mode state
        if not hasattr(self, 'pan_mode'):
            self.pan_mode = False

        self.ax.imshow(img)
        # Store original view limits for zoom reset
        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()
        self.zoom_level = 1.0
        
        # Enhanced title with progress
        progress_text = f"Image {self.current_idx + 1}/{len(self.image_files)} ({self.mode}): {img_path.name}"
        self.ax.set_title(progress_text)
        
        # Reset for new image
        self.bbox_coords = []
        self.current_boxes = []
        
        # Connect mouse events for drag functionality
        self.press_cid = self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.motion_cid = self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.release_cid = self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        
        # Connect zoom and pan events
        self.scroll_cid = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.key_cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Create class selection buttons (top row)
        button_width = 0.12
        start_x = 0.1
        
        self.class_buttons = {}
        for i, class_name in enumerate(self.classes):
            ax_class = plt.axes([start_x + i * (button_width + 0.02), 0.15, button_width, 0.05])
            btn = Button(ax_class, class_name.title())
            btn.on_clicked(lambda event, cls=class_name: self.set_class(cls))
            self.class_buttons[class_name] = btn
            
            # Highlight current class
            if class_name == self.current_class:
                btn.color = 'lightblue'
        
        # Create action buttons (bottom row)
        ax_skip = plt.axes([0.02, 0.05, 0.08, 0.06])
        ax_archive = plt.axes([0.12, 0.05, 0.08, 0.06])
        ax_save = plt.axes([0.22, 0.05, 0.08, 0.06])
        ax_next = plt.axes([0.32, 0.05, 0.08, 0.06])
        ax_clear = plt.axes([0.42, 0.05, 0.08, 0.06])
        ax_undo = plt.axes([0.52, 0.05, 0.08, 0.06])
        ax_reset = plt.axes([0.62, 0.05, 0.08, 0.06])
        ax_quit = plt.axes([0.72, 0.05, 0.08, 0.06])

        self.btn_skip = Button(ax_skip, 'Skip')
        self.btn_archive = Button(ax_archive, 'Archive')
        self.btn_save = Button(ax_save, 'Save')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_clear = Button(ax_clear, 'Clear All')
        self.btn_undo = Button(ax_undo, 'Undo Last')
        self.btn_reset = Button(ax_reset, 'Reset Zoom')
        self.btn_quit = Button(ax_quit, 'Quit')

        # Connect action button events
        self.btn_skip.on_clicked(self.skip_image)
        self.btn_archive.on_clicked(self.archive_current_image)
        self.btn_save.on_clicked(self.save_annotation)
        self.btn_next.on_clicked(self.next_image)
        self.btn_clear.on_clicked(self.clear_boxes)
        self.btn_undo.on_clicked(self.undo_last_box)
        self.btn_reset.on_clicked(self.reset_zoom)
        self.btn_quit.on_clicked(self.quit_annotation)


        # Show current class info
        self.class_text = self.fig.text(0.02, 0.95, f"Current: {self.current_class.title()}", 
                                       fontsize=14, weight='bold', 
                                       color=self.class_colors.get(self.current_class, 'black'))
        # Show zoom level indicator
        self.zoom_text = self.fig.text(0.02, 0.90, f"Zoom: {self.zoom_level:.1f}x", 
                                    fontsize=12, weight='bold', color='blue')
        
        plt.show()


    def on_scroll(self, event):
        """Handle mouse wheel zoom"""
        if event.inaxes != self.ax:
            return
        
        # Zoom factor and limits
        zoom_factor = 1.2 if event.step > 0 else 1/1.2
        max_zoom = 10.0
        min_zoom = 0.5
        
        # Calculate new zoom level
        new_zoom = self.zoom_level * zoom_factor
        if new_zoom > max_zoom or new_zoom < min_zoom:
            return  # Don't zoom beyond limits
        
        # Get current axis limits and mouse position
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        
        # Calculate zoom around mouse position
        xrange = xlim[1] - xlim[0]
        yrange = ylim[1] - ylim[0]
        
        # New range after zoom
        new_xrange = xrange / zoom_factor
        new_yrange = yrange / zoom_factor
        
        # Center new view on mouse position
        x_center_ratio = (xdata - xlim[0]) / xrange
        y_center_ratio = (ydata - ylim[0]) / yrange
        
        new_x1 = xdata - new_xrange * x_center_ratio
        new_x2 = xdata + new_xrange * (1 - x_center_ratio)
        new_y1 = ydata - new_yrange * y_center_ratio
        new_y2 = ydata + new_yrange * (1 - y_center_ratio)
        
        # Apply new limits
        self.ax.set_xlim(new_x1, new_x2)
        self.ax.set_ylim(new_y1, new_y2)
        
        # Update zoom level and indicator
        self.zoom_level = new_zoom
        self.zoom_text.set_text(f"Zoom: {self.zoom_level:.1f}x")
        
        try:
            plt.draw()
        except Exception as e:
            print(f"⚠️ Display update skipped: {e}")

    def on_key_press(self, event):
        """Handle keyboard events for panning and pan mode toggle"""
        if event.inaxes != self.ax:
            return
        
        # Toggle pan mode with 'p' key
        if event.key == 'p':
            self.toggle_pan_mode()
            return
        
        # Pan with arrow keys when zoomed in
        if self.zoom_level > 1.0:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            # Pan distance (10% of current view)
            x_pan = (xlim[1] - xlim[0]) * 0.1
            y_pan = (ylim[1] - ylim[0]) * 0.1
            
            if event.key == 'left':
                self.ax.set_xlim(xlim[0] - x_pan, xlim[1] - x_pan)
            elif event.key == 'right':
                self.ax.set_xlim(xlim[0] + x_pan, xlim[1] + x_pan)
            elif event.key == 'up':
                self.ax.set_ylim(ylim[0] + y_pan, ylim[1] + y_pan)
            elif event.key == 'down':
                self.ax.set_ylim(ylim[0] - y_pan, ylim[1] - y_pan)
            
            try:
                plt.draw()
            except Exception as e:
                print(f"⚠️ Display update skipped: {e}")

    def toggle_pan_mode(self):
        """Toggle pan mode on/off"""
        if not hasattr(self, 'pan_mode'):
            self.pan_mode = False
        
        self.pan_mode = not self.pan_mode
        
        # Update pan mode indicator
        if hasattr(self, 'pan_mode_text'):
            self.pan_mode_text.remove()
        
        if self.pan_mode:
            self.pan_mode_text = self.fig.text(0.02, 0.85, "PAN MODE: ON (Press 'P' to toggle)", 
                                            fontsize=12, weight='bold', color='orange',
                                            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
            print("🔄 Pan mode ON - box creation disabled (press 'P' to toggle)")
        else:
            self.pan_mode_text = self.fig.text(0.02, 0.85, "", fontsize=1)  # Hidden
            print("🔄 Pan mode OFF - box creation enabled")
        
        try:
            plt.draw()
        except Exception as e:
            print(f"⚠️ Display update skipped: {e}")

    def reset_zoom(self, event=None):
        """Reset zoom to original view"""
        if self.original_xlim is not None and self.original_ylim is not None:
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.zoom_level = 1.0
            self.zoom_text.set_text(f"Zoom: {self.zoom_level:.1f}x")
            try:
                plt.draw()
            except Exception as e:
                print(f"⚠️ Display update skipped: {e}")
            print("🔄 Zoom reset to original view")


    def skip_image(self, event=None):
        """Skip current image without saving"""
        print("⏭️ Skipped image")
        self.current_idx += 1
        self.show_image()


    def archive_current_image(self, event=None):
        """Archive current image (move to archived_images folder) and continue to next"""
        if self.current_idx >= len(self.image_files):
            print("⚠️ No image to archive")
            return
        
        current_image = self.image_files[self.current_idx]
        
        # Create archive directory
        archive_dir = self.images_dir.parent / 'archived_images'
        archive_dir.mkdir(exist_ok=True)
        
        # Move image to archive
        archived_path = archive_dir / current_image.name
        
        try:
            shutil.move(str(current_image), str(archived_path))
            
            # Update annotation status
            self.update_annotation_status_archived(current_image.name)
            
            # Remove from current session
            self.image_files.pop(self.current_idx)
            
            # Adjust index if we're at the end
            if self.current_idx >= len(self.image_files):
                self.current_idx = len(self.image_files) - 1
            
            print(f"📦 Archived: {current_image.name}")
            
            # Show next image or complete if done
            if len(self.image_files) > 0:
                self.show_image()
            else:
                print("🎉 All images in current mode completed!")
                plt.close('all')
                
        except Exception as e:
            print(f"❌ Failed to archive image: {e}")


    def quit_annotation(self, event=None):
        """Save current work and quit annotation session"""
        # Save current image if it has annotations
        if self.current_idx < len(self.image_files):
            img_path = self.image_files[self.current_idx]
            img_name = img_path.stem
            
            if img_name in self.annotations and self.annotations[img_name]:
                self.save_annotation()
                print("💾 Current image saved before quitting")
        
        # Generate comprehensive session summary
        self.print_session_summary()
        
        plt.close('all')

    def print_session_summary(self):
        """Generate user-friendly session summary with recommendations"""
        stats = self.get_dataset_statistics()
        completed_this_session = self.current_idx
        remaining_in_mode = len(self.image_files) - self.current_idx if self.current_idx < len(self.image_files) else 0
        
        print("\n" + "="*60)
        print("📋 ANNOTATION SESSION COMPLETED")
        print("="*60)
        
        # Session Results
        print(f"⏰ Session Results:")
        print(f"   • Images processed this session: {completed_this_session}")
        print(f"   • Images remaining in '{self.mode}' mode: {remaining_in_mode}")
        print(f"   • Total dataset progress: {stats['annotated']}/{stats['total']} ({stats['percentage']:.1f}%)")
        
        if stats['archived'] > 0:
            print(f"   • Images archived: {stats['archived']}")
        
        # Smart recommendations
        recommendations = self.get_annotation_recommendations(stats)
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   • {rec}")
        
        # Next steps
        next_steps = self.get_next_steps(stats)
        if next_steps:
            print(f"\n🎯 Suggested Next Steps:")
            for step in next_steps:
                print(f"   • {step}")
        
        # Files updated
        print(f"\n📁 Files Updated:")
        print(f"   • Annotation status tracking (.annotation_status.json)")
        if stats['archived'] > 0:
            print(f"   • Archived images moved to archived_images/ folder")
        
        print("="*60)
        
        # Update project status file
        self.update_project_status_file(stats)

    def get_annotation_recommendations(self, stats):
        """Simple rule-based annotation recommendations"""
        recommendations = []
        
        # Based on dataset size
        if stats['annotated'] < 50:
            recommendations.append("Aim for 50+ annotations before first training attempt")
        elif stats['annotated'] < 100:
            recommendations.append("Getting close! 100+ annotations will give better results")
        elif stats['annotated'] < 200:
            recommendations.append("Good progress! Consider data augmentation to multiply dataset")
        else:
            recommendations.append("Excellent dataset size! Ready for professional training")
        
        # Based on completion percentage  
        if stats['percentage'] < 80 and stats['unannotated'] > 20:
            recommendations.append(f"Continue annotating - {stats['unannotated']} images remaining")
        elif stats['unannotated'] > 0:
            recommendations.append("Almost done! Just a few more images to complete")
        
        # Based on archived images
        if stats['archived'] > stats['annotated'] * 0.3:  # If >30% archived
            recommendations.append("High archive rate - consider collecting higher quality images")
        
        return recommendations

    def get_next_steps(self, stats):
        """Simple workflow recommendations"""
        next_steps = []
        
        if stats['annotated'] >= 50 and stats['unannotated'] == 0:
            # Ready for full pipeline
            next_steps.append("Run yolo_conversion.py (convert annotations to YOLO format)")
            next_steps.append("Run data_augmentation.py (multiply dataset 3-5x)")
            next_steps.append("Run dataset_splitting.py (prepare train/validation sets)")
            next_steps.append("Run model_training.py (train your detection model)")
            
        elif stats['annotated'] >= 50 and stats['unannotated'] > 0:
            # Can start pipeline or continue annotating
            next_steps.append("Option 1: Continue annotating remaining images")
            next_steps.append("Option 2: Proceed with current annotations for initial training")
            next_steps.append("Recommended: At least finish current batch before training")
            
        elif stats['annotated'] >= 20:
            # Getting close to minimum viable dataset
            next_steps.append("Continue annotating - aim for 50+ before first training")
            next_steps.append("Focus on variety: different lighting, angles, backgrounds")
            
        else:
            # Still need more annotations
            next_steps.append("Continue annotation sessions - need more examples for training")
            next_steps.append("Target: 50+ annotations for basic model, 100+ for good results")
            next_steps.append("Tip: Annotate diverse examples (different conditions)")
        
        return next_steps

    def update_project_status_file(self, stats):
        """Create/update simple project status dashboard"""
        try:
            dataset_dir = self.images_dir.parent
            status_file = dataset_dir / 'PROJECT_STATUS.txt'
            
            # Get recent activity
            recent_activity = self.get_recent_activity()
            
            # Generate status content
            status_content = f"""DOG POOP DETECTION PROJECT STATUS
    Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

    DATASET STATUS:
    ├── Images Collected: {stats['total']}
    ├── Images Annotated: {stats['annotated']} ({stats['percentage']:.1f}%)
    ├── Images Archived: {stats['archived']}
    ├── Images Remaining: {stats['unannotated']}
    └── Ready for Training: {'YES' if stats['annotated'] >= 50 else 'NO'}

    MODEL STATUS:
    ├── Training Data Available: {'YES' if stats['annotated'] >= 50 else 'NO (Need 50+)'}
    ├── Augmentation Ready: {'YES' if stats['annotated'] >= 20 else 'NO (Need 20+)'}
    └── Production Ready: {'YES' if stats['annotated'] >= 100 else 'NO (Need 100+)'}

    RECENT ACTIVITY:
    {recent_activity}

    RECOMMENDATIONS:
    {self.get_status_recommendations(stats)}

    NEXT ACTIONS:
    {self.get_status_next_actions(stats)}
    """
            
            # Write with explicit UTF-8 encoding
            with open(status_file, 'w', encoding='utf-8') as f:
                f.write(status_content)
                
            print(f"📄 Updated project status: {status_file}")
            
        except Exception as e:
            print(f"⚠️ Could not update project status file: {e}")
            # Try fallback with ASCII-only content
            try:
                simple_content = f"""PROJECT STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M')}
    Annotated: {stats['annotated']}/{stats['total']} ({stats['percentage']:.1f}%)
    Archived: {stats['archived']}
    Ready for Training: {'YES' if stats['annotated'] >= 50 else 'NO'}
    """
                with open(status_file, 'w', encoding='ascii', errors='replace') as f:
                    f.write(simple_content)
                print(f"📄 Created simplified project status (ASCII mode)")
            except Exception as e2:
                print(f"⚠️ Status file creation failed completely: {e2}")

    def get_recent_activity(self):
        """Get recent activity summary"""
        try:
            dataset_dir = self.images_dir.parent
            
            # Check extraction log
            extraction_log = dataset_dir / '.extraction_log.json'
            recent_extractions = "No recent extractions"
            if extraction_log.exists():
                with open(extraction_log, 'r') as f:
                    log_data = json.load(f)
                if log_data:
                    latest_session = max(log_data.keys())
                    latest_info = log_data[latest_session]
                    frames = latest_info.get('frames_extracted', 0)
                    date = latest_info.get('date', '')[:10] 
                    recent_extractions = f"{date}: Extracted {frames} new images"
            
            # Current annotation session
            session_info = f"Today: Annotated {self.current_idx} images"
            
            return f"- {recent_extractions}\n- {session_info}"
            
        except:
            return "- Recent extraction activity\n- Current annotation session"

    def get_status_recommendations(self, stats):
        """Get recommendations for status file"""
        if stats['annotated'] >= 100:
            return "Dataset is excellent! Ready for advanced training and production use."
        elif stats['annotated'] >= 50:
            return "Dataset is good! Consider augmentation and initial model training."
        elif stats['annotated'] >= 20:
            return "Keep annotating! You're close to a trainable dataset."
        else:
            return "Continue collecting and annotating more examples."

    def get_status_next_actions(self, stats):
        """Get next actions for status file"""
        if stats['unannotated'] == 0 and stats['annotated'] >= 50:
            return "1. Run yolo_conversion.py\n2. Run data_augmentation.py\n3. Train first model"
        elif stats['annotated'] >= 50:
            return "1. Finish annotating remaining images\n2. Or proceed with current data for initial training"
        else:
            return f"1. Continue annotation sessions\n2. Target: {50 - stats['annotated']} more annotations needed"


    def set_class(self, class_name):
        """Set the current annotation class"""
        self.current_class = class_name
        
        # Update button colors
        for cls, btn in self.class_buttons.items():
            if cls == class_name:
                btn.color = 'lightblue'
            else:
                btn.color = 'white'
        
        # Update class text
        self.class_text.set_text(f"Current: {class_name.title()}")
        self.class_text.set_color(self.class_colors.get(class_name, 'black'))
        
        try:
            plt.draw()
        except Exception as e:
            print(f"⚠️ Display update skipped: {e}")
        print(f"🏷️ Selected class: {class_name}")


    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two boxes"""
        # Box coordinates
        x1_1, y1_1 = box1['x'], box1['y']
        x2_1, y2_1 = x1_1 + box1['width'], y1_1 + box1['height']
        
        x1_2, y1_2 = box2['x'], box2['y']
        x2_2, y2_2 = x1_2 + box2['width'], y1_2 + box2['height']
        
        # Intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Union area
        area1 = box1['width'] * box1['height']
        area2 = box2['width'] * box2['height']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def check_overlap(self, new_box, existing_boxes, threshold=0.1):
        """Check if new box significantly overlaps with existing boxes"""
        overlaps = []
        for existing_box in existing_boxes:
            iou = self.calculate_iou(new_box, existing_box)
            if iou > threshold:
                overlaps.append({
                    'box': existing_box,
                    'iou': iou,
                    'same_class': existing_box['label'] == new_box['label']
                })
        
        return overlaps

    def on_mouse_press(self, event):
        """Handle mouse press - start drag (if not in pan mode)"""
        if event.inaxes != self.ax or event.button != 1:  # Only left mouse button
            return
        
        # Check if pan mode is active
        if hasattr(self, 'pan_mode') and self.pan_mode:
            # In pan mode - handle panning instead of box creation
            self.is_panning = True
            self.pan_start = (event.xdata, event.ydata)
            print("🔄 Panning mode - click and drag to pan")
            return
        
        # Normal box creation mode
        self.is_dragging = True
        self.drag_start = (event.xdata, event.ydata)
        
        print(f"📍 Starting {self.current_class} box at ({event.xdata:.0f}, {event.ydata:.0f})")

    def on_mouse_move(self, event):
        """Handle mouse move - update preview or pan"""
        if event.inaxes != self.ax:
            return
        
        # Handle panning
        if hasattr(self, 'is_panning') and self.is_panning and hasattr(self, 'pan_start') and self.pan_start:
            # Calculate pan offset
            x_start, y_start = self.pan_start
            x_offset = event.xdata - x_start
            y_offset = event.ydata - y_start
            
            # Apply pan
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            self.ax.set_xlim(xlim[0] - x_offset, xlim[1] - x_offset)
            self.ax.set_ylim(ylim[0] - y_offset, ylim[1] - y_offset)
            
            try:
                plt.draw()
            except Exception as e:
                print(f"⚠️ Display update skipped: {e}")
            return
        
        # Handle box preview (normal annotation mode)
        if not self.is_dragging or self.drag_start is None:
            return
        
        # Constrain to image bounds
        img_height, img_width = self.ax.images[0].get_array().shape[:2]
        x_current = max(0, min(event.xdata, img_width))
        y_current = max(0, min(event.ydata, img_height))
        
        # Calculate rectangle
        x1, y1 = self.drag_start
        x = min(x1, x_current)
        y = min(y1, y_current)
        width = abs(x_current - x1)
        height = abs(y_current - y1)
        
        # Remove previous preview rectangle
        if self.preview_rect is not None:
            self.preview_rect.remove()
        
        # Create new preview rectangle (dashed)
        color = self.class_colors.get(self.current_class, 'red')
        self.preview_rect = patches.Rectangle((x, y), width, height, 
                                            linewidth=2, edgecolor=color, 
                                            facecolor='none', linestyle='--', alpha=0.7)
        self.ax.add_patch(self.preview_rect)
        
        try:
            plt.draw()
        except Exception as e:
            print(f"⚠️ Display update skipped: {e}")

    def on_mouse_release(self, event):
        """Handle mouse release - finalize box or end pan"""
        if event.inaxes != self.ax:
            return
        
        # Handle end of panning
        if hasattr(self, 'is_panning') and self.is_panning:
            self.is_panning = False
            self.pan_start = None
            print("🔄 Pan completed")
            return
        
        # Normal box creation logic (existing code)
        if not self.is_dragging or self.drag_start is None:
            return
        
        # Auto-save periodically
        if hasattr(self, 'boxes_since_save'):
            self.boxes_since_save += 1
        else:
            self.boxes_since_save = 1
        
        if self.boxes_since_save >= 5:  # Auto-save every 5 boxes
            self.save_annotation()
            self.boxes_since_save = 0
            print("💾 Auto-saved (5 boxes completed)")

        # Constrain to image bounds
        img_height, img_width = self.ax.images[0].get_array().shape[:2]
        x_current = max(0, min(event.xdata, img_width))
        y_current = max(0, min(event.ydata, img_height))
        
        # Calculate final rectangle
        x1, y1 = self.drag_start
        x = min(x1, x_current)
        y = min(y1, y_current)
        width = abs(x_current - x1)
        height = abs(y_current - y1)
        
        # Remove preview rectangle
        if self.preview_rect is not None:
            self.preview_rect.remove()
            self.preview_rect = None
        
        # Handle very small boxes
        if width < 10 or height < 10:
            width = max(20, width)
            height = max(20, height)
            print(f"📦 Created minimum size box ({width}x{height})")
        
        # Create new box annotation
        new_annotation = {
            'x': x, 'y': y, 'width': width, 'height': height,
            'label': self.current_class
        }

        # Check for overlaps
        img_name = self.image_files[self.current_idx].stem
        existing_annotations = self.annotations.get(img_name, [])
        overlaps = self.check_overlap(new_annotation, existing_annotations)

        # Handle overlaps with non-blocking approach
        if overlaps:
            self.handle_overlap_situation(new_annotation, overlaps, x, y, width, height)
        else:
            self.finalize_annotation(new_annotation, x, y, width, height)
        
        # Reset drag state
        self.is_dragging = False
        self.drag_start = None

    def handle_overlap_situation(self, new_annotation, overlaps, x, y, width, height):
        """Handle overlap situation with non-blocking GUI approach"""
        # Calculate overlap details
        overlap_warnings = []
        max_overlap = 0
        for overlap in overlaps:
            same_class_str = "same class" if overlap['same_class'] else "different class"
            overlap_warnings.append(f"{overlap['iou']:.1%} with {overlap['box']['label']} ({same_class_str})")
            max_overlap = max(max_overlap, overlap['iou'])
        
        warning_msg = f"⚠️ Overlap detected: {', '.join(overlap_warnings)}"
        print(warning_msg)
        
        # For high overlaps (>50%), show confirmation dialog
        if max_overlap > 0.5:
            self.show_overlap_confirmation_dialog(new_annotation, max_overlap, x, y, width, height)
        else:
            # Low overlap - proceed with warning only
            print("ℹ️ Low overlap detected - proceeding automatically")
            self.finalize_annotation(new_annotation, x, y, width, height)

    def show_overlap_confirmation_dialog(self, new_annotation, overlap_percentage, x, y, width, height):
        """Show non-blocking confirmation dialog for high overlaps"""
        # Clean up any existing dialog first
        if hasattr(self, 'overlap_dialog_elements'):
            self.cleanup_overlap_dialog()
        
        # Store the pending annotation for the dialog callbacks
        self.pending_annotation = {
            'annotation': new_annotation,
            'x': x, 'y': y, 'width': width, 'height': height
        }
        
        # Create confirmation dialog buttons
        dialog_y = 0.85 
        
        # Warning text
        warning_text = f"High overlap detected ({overlap_percentage:.1%})"
        self.overlap_warning_text = self.fig.text(0.5, dialog_y + 0.08, warning_text, 
                                                ha='center', fontsize=12, weight='bold', 
                                                color='red', bbox=dict(boxstyle="round,pad=0.3", 
                                                facecolor='yellow', alpha=0.8))
        
        # Confirmation buttons
        ax_confirm = plt.axes([0.35, dialog_y, 0.12, 0.05])
        ax_cancel = plt.axes([0.53, dialog_y, 0.12, 0.05])
        
        self.btn_confirm_overlap = Button(ax_confirm, 'Keep Box')
        self.btn_cancel_overlap = Button(ax_cancel, 'Cancel Box')
        
        # Set button colors
        self.btn_confirm_overlap.color = 'lightgreen'
        self.btn_cancel_overlap.color = 'lightcoral'
        
        # Connect button events with error handling
        try:
            self.btn_confirm_overlap.on_clicked(self.confirm_overlap_box)
            self.btn_cancel_overlap.on_clicked(self.cancel_overlap_box)
        except Exception as e:
            print(f"⚠️ Button connection error: {e}")
        
        # Store references for cleanup
        self.overlap_dialog_elements = [
            self.overlap_warning_text, 
            ax_confirm,
            ax_cancel
        ]
        
        try:
            plt.draw()
        except Exception as e:
            print(f"⚠️ Drawing error: {e}")
        
        print(f"🔔 High overlap ({overlap_percentage:.1%}) - please confirm using buttons")

    def confirm_overlap_box(self, event=None):
        """User confirmed to keep the overlapping box"""
        if hasattr(self, 'pending_annotation'):
            annotation = self.pending_annotation['annotation']
            x = self.pending_annotation['x']
            y = self.pending_annotation['y'] 
            width = self.pending_annotation['width']
            height = self.pending_annotation['height']
            
            print("✅ User confirmed - keeping overlapping box")
            self.finalize_annotation(annotation, x, y, width, height)
            
        self.cleanup_overlap_dialog()

    def cancel_overlap_box(self, event=None):
        """User canceled the overlapping box"""
        print("❌ User canceled - removing overlapping box")
        
        # Don't finalize the annotation, just clean up
        self.cleanup_overlap_dialog()

    def cleanup_overlap_dialog(self):
        """Clean up overlap confirmation dialog elements"""
        if hasattr(self, 'overlap_dialog_elements'):
            # Disconnect button events first
            if hasattr(self, 'btn_confirm_overlap'):
                try:
                    self.btn_confirm_overlap.disconnect_events()
                except:
                    pass
            
            if hasattr(self, 'btn_cancel_overlap'):
                try:
                    self.btn_cancel_overlap.disconnect_events()
                except:
                    pass
            
            # Remove visual elements
            for element in self.overlap_dialog_elements:
                try:
                    if hasattr(element, 'remove'):
                        element.remove()
                    elif hasattr(element, 'set_visible'):
                        element.set_visible(False)
                except:
                    pass  # Element might already be removed
            
            # Clear references
            delattr(self, 'overlap_dialog_elements')
            
            if hasattr(self, 'btn_confirm_overlap'):
                delattr(self, 'btn_confirm_overlap')
            if hasattr(self, 'btn_cancel_overlap'):
                delattr(self, 'btn_cancel_overlap')
        
        if hasattr(self, 'pending_annotation'):
            delattr(self, 'pending_annotation')
        
        # Use safer drawing method
        try:
            plt.draw()
        except Exception as e:
            print(f"⚠️ Display update skipped: {e}")

    def finalize_annotation(self, new_annotation, x, y, width, height):
        """Finalize and store the annotation"""
        # Get color for current class
        color = self.class_colors.get(self.current_class, 'red')
        
        # Create final rectangle (solid line)
        rect = patches.Rectangle((x, y), width, height, 
                            linewidth=2, edgecolor=color, facecolor='none')
        self.ax.add_patch(rect)
        
        # Add class label near the box
        self.ax.text(x, y-10, self.current_class.upper(), 
                    color=color, fontsize=10, weight='bold')
        
        self.current_boxes.append({'rect': rect, 'class': self.current_class})
        
        # Store annotation
        img_name = self.image_files[self.current_idx].stem
        if img_name not in self.annotations:
            self.annotations[img_name] = []
        
        self.annotations[img_name].append(new_annotation)
        
        print(f"📦 {self.current_class.upper()} box added: ({x:.0f}, {y:.0f}, {width:.0f}×{height:.0f})")
        
        try:
            plt.draw()
        except Exception as e:
            print(f"⚠️ Display update skipped: {e}")
    
    def undo_last_box(self, event=None):
        """Remove the last added bounding box"""
        img_name = self.image_files[self.current_idx].stem
        
        if img_name in self.annotations and self.annotations[img_name]:
            # Remove from annotations
            removed_annotation = self.annotations[img_name].pop()
            
            # Remove visual box
            if self.current_boxes:
                box_info = self.current_boxes.pop()
                box_info['rect'].remove()

            try:
                plt.draw()
            except Exception as e:
                print(f"⚠️ Display update skipped: {e}")

            print(f"↩️ Removed last {removed_annotation['label']} box")
        else:
            print("⚠️ No boxes to undo")
    
    def clear_boxes(self, event=None):
        """Clear all boxes"""
        img_name = self.image_files[self.current_idx].stem
        if img_name in self.annotations:
            del self.annotations[img_name]
        
        for box_info in self.current_boxes:
            box_info['rect'].remove()
        self.current_boxes = []
        try:
            plt.draw()
        except Exception as e:
            print(f"⚠️ Display update skipped: {e}")
        print("🗑️ Cleared all boxes")
    
    def save_annotation(self, event=None):
        """Save current annotations and update status"""
        img_path = self.image_files[self.current_idx]
        img_name = img_path.stem
        
        if img_name in self.annotations:
            annotation_data = {
                'imagePath': img_path.name,
                'shapes': [
                    {
                        'label': ann['label'],
                        'points': [
                            [ann['x'], ann['y']],
                            [ann['x'] + ann['width'], ann['y'] + ann['height']]
                        ],
                        'shape_type': 'rectangle'
                    }
                    for ann in self.annotations[img_name]
                ]
            }
            
            json_path = self.labels_dir / f"{img_name}.json"
            with open(json_path, 'w') as f:
                json.dump(annotation_data, f, indent=2)
            
            # Update annotation status
            self.update_annotation_status(img_path.name)
            
            # Count annotations by class
            class_counts = {}
            for ann in self.annotations[img_name]:
                class_name = ann['label']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            count_str = ', '.join([f"{count} {cls}" for cls, count in class_counts.items()])
            print(f"💾 Saved: {count_str}")
        else:
            print("⚠️ No annotations to save")
    
    def next_image(self, event=None):
        """Save and go to next image"""
        self.save_annotation()
        self.current_idx += 1
        self.show_image()

def show_mode_selection_menu():
    """Display mode selection menu and return chosen mode"""
    
    # Get dataset statistics
    images_dir = Path(config.get_dataset_path('images'))
    dataset_dir = images_dir.parent
    status_file = dataset_dir / '.annotation_status.json'
    
    # Load status
    if status_file.exists():
        with open(status_file, 'r') as f:
            annotation_status = json.load(f)
    else:
        annotation_status = {}
    
    # Calculate statistics
    all_images = list(images_dir.glob("*.jpg"))
    total = len(all_images)
    
    annotated_count = 0
    new_count = 0
    
    for img_path in all_images:
        img_name = img_path.name
        status = annotation_status.get(img_name, {})
        if status.get('status', 'unannotated') == 'annotated':
            annotated_count += 1
        else:
            new_count += 1
    
    print(get_project_title('annotation'))
    print("=" * 40)
    print("Dataset Status:")
    print(f"  • Total images: {total}")
    print(f"  • Annotated: {annotated_count} ({annotated_count/total*100:.1f}%)" if total > 0 else "  • Annotated: 0")
    print(f"  • Unannotated: {new_count}")
    
    # Show last session info if available
    extraction_log = dataset_dir / '.extraction_log.json'
    if extraction_log.exists():
        with open(extraction_log, 'r') as f:
            log_data = json.load(f)
        if log_data:
            latest_session = max(log_data.keys())
            latest_info = log_data[latest_session]
            print(f"  • Last session: {latest_info.get('frames_extracted', 0)} new images")
    
    print()
    print("Mode Selection:")
    print(f"  1. New images only ({new_count} images) [DEFAULT]")
    print(f"  2. All images ({total} images)")
    print(f"  3. Resume unannotated ({new_count} images)")
    print()
    
    while True:
        choice = input("Choice (1-3, Enter for default): ").strip()
        
        if choice == '' or choice == '1':
            return 'new_only'
        elif choice == '2':
            return 'all'
        elif choice == '3':
            return 'resume'
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    print(get_project_title('annotation'))
    print("=" * 40)
    print("Instructions:")
    print("• Select class with top buttons")
    print("• Click and drag to create bounding box")
    print("• Press 'P' to toggle pan mode (disables box creation)")
    print("• Mouse wheel to zoom, arrow keys to pan when zoomed")
    print("• Different classes show in different colors")
    print("• Use action buttons to navigate")
    print("=" * 40)
    print()
    
    # Show mode selection menu
    mode = show_mode_selection_menu()
    
    # Get classes from config
    classes = config.get('dataset.classes', ['poop', 'leaf', 'stick', 'rock'])
    
    # Get paths
    images_dir = config.get_dataset_path('images')
    labels_dir = config.get_dataset_path('labels')
    
    if not Path(images_dir).exists():
        print(f"❌ Images directory not found: {images_dir}")
        exit()
    
    print(f"\n🚀 Starting annotation in '{mode}' mode...")
    print("=" * 40)
    
    annotator = MultiClassAnnotator(images_dir, labels_dir, classes, mode)
    annotator.start_annotation()