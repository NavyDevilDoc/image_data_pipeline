import json
import yaml
from pathlib import Path
from datetime import datetime
import matplotlib.colors as mcolors

def generate_class_colors(classes):
    """
    Auto-generate distinct colors for any class list
    
    Args:
        classes: List of class names
        
    Returns:
        Dictionary mapping class names to colors
    """
    # Use matplotlib's tableau colors for consistency and visual appeal
    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    # If more classes than base colors, extend with CSS4 colors
    if len(classes) > len(color_palette):
        extended_colors = list(mcolors.CSS4_COLORS.values())
        # Filter out very light colors that won't show well
        extended_colors = [c for c in extended_colors if not any(
            light in c.lower() for light in ['white', 'light', 'pale', 'ivory', 'snow']
        )]
        color_palette.extend(extended_colors)
    
    return {cls: color_palette[i % len(color_palette)] 
            for i, cls in enumerate(classes)}


def get_project_title(context='general'):
    """
    Get context-specific project title
    
    Args:
        context: Context for the title ('extraction', 'annotation', 'training', etc.)
        
    Returns:
        Formatted project title string
    """
    project_name = config.get('project.name', 'detection_project')
    description = config.get('project.description', 'Multi-class detection system')
    
    if context == 'extraction':
        return f"🎯 {description.title()} - Frame Extraction"
    elif context == 'annotation':
        return f"🏷️ {description.title()} - Annotation Tool"
    elif context == 'training':
        return f"🤖 {description.title()} - Model Training"
    else:
        return f"🎯 {description.title()}"

class ProjectConfig:
    """
    Centralized configuration management for the poop detection pipeline
    """
    
    def __init__(self, config_file="project_config.yaml"):
        self.config_file = Path(config_file)
        self.config = self.load_or_create_config()
    
    def load_or_create_config(self):
        """Load existing config or create default"""
        
        if self.config_file.exists():
            print(f"📄 Loading config from: {self.config_file}")
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            print(f"📄 Creating default config: {self.config_file}")
            return self.create_default_config()
    
    def create_default_config(self):
        """Create default configuration"""
        
        default_config = {
            # Project metadata
            'project': {
                'name': 'poop_detection',
                'version': '1.0.0',
                'created': datetime.now().isoformat(),
                'description': 'Multi-class dog poop detection system'
            },
            
            # Dataset configuration
            'dataset': {
                'name': 'poop_detection_dataset',
                'classes': ['poop', 'leaf', 'stick', 'rock'],
                'class_colors': {
                    'poop': 'red',
                    'leaf': 'green', 
                    'stick': 'brown',
                    'rock': 'gray'
                }
            },
            
            # Frame extraction settings
            'extraction': {
                'target_frames_per_video': 25,
                'output_format': 'jpg',
                'quality': 95,
                'preview_samples': 6
            },
            
            # Annotation settings
            'annotation': {
                'default_class': 'poop',
                'auto_save': True,
                'validation_on_save': True
            },
            
            # YOLO conversion settings
            'yolo': {
                'format_version': '8',
                'coordinate_format': 'normalized',
                'min_annotation_rate': 0.2  # 20% minimum
            },
            
            # Dataset splitting
            'splitting': {
                'train_ratio': 0.8,
                'val_ratio': 0.2,
                'test_ratio': 0.0,
                'random_seed': 42,
                'stratified': False
            },
            
            # Directory structure
            'directories': {
                'base': 'poop_detection_dataset',
                'images': 'images',
                'labels': 'labels', 
                'yolo_labels': 'yolo_labels',
                'train': 'train',
                'val': 'val',
                'test': 'test'
            },
            
            # File patterns
            'patterns': {
                'image_extensions': ['.jpg', '.jpeg', '.png'],
                'video_extensions': ['.mp4', '.avi', '.mov', '.mkv'],
                'timestamp_format': '%Y%m%d_%H%M%S'
            }
        }
        
        self.save_config(default_config)
        return default_config
    
    def save_config(self, config=None):
        """Save configuration to file"""
        
        if config is None:
            config = self.config
            
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"💾 Configuration saved to: {self.config_file}")
    
    def get(self, key_path, default=None):
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path (e.g., 'dataset.classes')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path, value):
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Dot-separated path (e.g., 'dataset.classes')
            value: Value to set
        """
        
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set value
        config[keys[-1]] = value
        self.save_config()
    
    def get_dataset_path(self, subdir=None):
        """Get full path to dataset directory or subdirectory"""
        
        base_path = Path(self.get('directories.base'))
        
        if subdir:
            return base_path / self.get(f'directories.{subdir}', subdir)
        else:
            return base_path

    def validate_config(self):
        """Validate configuration completeness"""
        
        required_keys = [
            'dataset.classes',
            'directories.base',
            'splitting.train_ratio',
            'extraction.target_frames_per_video'
        ]
        
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"⚠️ Missing configuration keys: {missing_keys}")
            return False
        
        print("✅ Configuration validation passed")
        return True

    def generate_dynamic_colors(self, classes=None):
        """
        Generate or update class colors dynamically
        
        Args:
            classes: List of classes (uses config classes if None)
            
        Returns:
            Dictionary of class colors
        """
        if classes is None:
            classes = self.get('dataset.classes', [])
        
        # Check if colors already exist and match class count
        existing_colors = self.get('dataset.class_colors', {})
        
        if len(existing_colors) == len(classes) and all(cls in existing_colors for cls in classes):
            return existing_colors
        
        # Generate new colors
        new_colors = generate_class_colors(classes)
        
        # Update config
        self.set('dataset.class_colors', new_colors)
        
        print(f"🎨 Generated colors for {len(classes)} classes")
        return new_colors

    def create_project_template(self, project_name, classes, description="Multi-class detection system"):
        """
        Create a new project configuration
        
        Args:
            project_name: Name of the new project
            classes: List of class names
            description: Project description
            
        Returns:
            New config dictionary
        """
        # Generate colors for the classes
        class_colors = generate_class_colors(classes)
        
        template_config = {
            'project': {
                'name': project_name,
                'version': '1.0.0',
                'created': datetime.now().isoformat(),
                'description': description
            },
            'dataset': {
                'name': f'{project_name}_dataset',
                'classes': classes,
                'class_colors': class_colors
            },
            # Copy all other settings from current config
            'extraction': self.get('extraction', {}),
            'annotation': self.get('annotation', {}),
            'yolo': self.get('yolo', {}),
            'splitting': self.get('splitting', {}),
            'directories': {
                'base': f'{project_name}_dataset',
                'images': 'images',
                'labels': 'labels', 
                'yolo_labels': 'yolo_labels',
                'train': 'train',
                'val': 'val',
                'test': 'test'
            },
            'patterns': self.get('patterns', {})
        }
        
        return template_config

    def print_summary(self):
        """Print configuration summary"""
        
        print("\n📋 PROJECT CONFIGURATION SUMMARY")
        print("=" * 50)
        print(f"Project: {self.get('project.name')} v{self.get('project.version')}")
        print(f"Dataset: {self.get('dataset.name')}")
        print(f"Classes: {', '.join(self.get('dataset.classes'))}")
        print(f"Base Directory: {self.get('directories.base')}")
        print(f"Train/Val Split: {self.get('splitting.train_ratio'):.0%}/{self.get('splitting.val_ratio'):.0%}")
        print(f"Target Frames: {self.get('extraction.target_frames_per_video')}")
        print("=" * 50)

# Global config instance
config = ProjectConfig()

if __name__ == "__main__":
    # Interactive configuration setup
    print("🎯 Poop Detection Project Configuration")
    print("=" * 50)
    
    config.print_summary()
    
    print("\n🔧 Configuration Options:")
    print("1. Update classes")
    print("2. Change dataset directory")
    print("3. Modify train/val split")
    print("4. Update frame extraction settings")
    print("5. Save current config")
    print("6. Exit")
    
    choice = input("\nChoice (1-6): ").strip()
    
    if choice == "1":
        current_classes = config.get('dataset.classes')
        print(f"Current classes: {current_classes}")
        new_classes = input("Enter new classes (comma-separated): ").strip()
        if new_classes:
            classes_list = [cls.strip() for cls in new_classes.split(',')]
            config.set('dataset.classes', classes_list)
            print(f"✅ Updated classes: {classes_list}")
    
    elif choice == "2":
        current_dir = config.get('directories.base')
        print(f"Current dataset directory: {current_dir}")
        new_dir = input("Enter new directory: ").strip()
        if new_dir:
            config.set('directories.base', new_dir)
            print(f"✅ Updated directory: {new_dir}")
    
    elif choice == "3":
        current_train = config.get('splitting.train_ratio')
        current_val = config.get('splitting.val_ratio')
        print(f"Current split - Train: {current_train:.0%}, Val: {current_val:.0%}")
        
        try:
            new_train = float(input("Enter train ratio (0.0-1.0): "))
            new_val = 1.0 - new_train
            
            config.set('splitting.train_ratio', new_train)
            config.set('splitting.val_ratio', new_val)
            print(f"✅ Updated split - Train: {new_train:.0%}, Val: {new_val:.0%}")
        except:
            print("❌ Invalid input")
    
    elif choice == "4":
        current_frames = config.get('extraction.target_frames_per_video')
        print(f"Current target frames: {current_frames}")
        
        try:
            new_frames = int(input("Enter new target frames: "))
            config.set('extraction.target_frames_per_video', new_frames)
            print(f"✅ Updated target frames: {new_frames}")
        except:
            print("❌ Invalid input")
    
    elif choice == "5":
        config.save_config()
        print("✅ Configuration saved!")
    
    print("\n👋 Configuration manager complete!")