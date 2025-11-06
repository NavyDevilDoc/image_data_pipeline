"""
Project Creation Wizard for Multi-Class Detection Systems
"""

import yaml
from pathlib import Path
from config import ProjectConfig, generate_class_colors

def create_new_project():
    """Interactive project creation wizard"""
    
    print("🏗️ Multi-Class Detection Project Creator")
    print("=" * 50)
    
    # Get project details
    project_name = input("Project name (e.g., 'sports_ball_detection'): ").strip()
    if not project_name:
        print("❌ Project name is required!")
        return
    
    description = input("Project description (e.g., 'Detect different sports balls'): ").strip()
    if not description:
        description = "Multi-class detection system"
    
    classes_input = input("Classes (comma-separated, e.g., 'baseball,football,basketball'): ").strip()
    if not classes_input:
        print("❌ At least one class is required!")
        return
    
    classes = [cls.strip() for cls in classes_input.split(',')]
    
    # Optional dataset directory name
    dataset_dir = input(f"Dataset directory name (default: '{project_name}_dataset'): ").strip()
    if not dataset_dir:
        dataset_dir = f'{project_name}_dataset'
    
    print(f"\n📋 Project Summary:")
    print(f"   • Name: {project_name}")
    print(f"   • Description: {description}")
    print(f"   • Classes: {', '.join(classes)} ({len(classes)} total)")
    print(f"   • Dataset directory: {dataset_dir}")
    
    # Auto-generate colors preview
    colors = generate_class_colors(classes)
    print(f"   • Colors: {', '.join([f'{cls}={color}' for cls, color in colors.items()])}")
    
    confirm = input("\nCreate this project? (y/n): ").lower().strip()
    if confirm != 'y':
        print("❌ Project creation cancelled")
        return
    
    # Create project config
    config_file = f"{project_name}_config.yaml"
    temp_config = ProjectConfig()  # Use existing config as template
    
    project_config = temp_config.create_project_template(project_name, classes, description)
    project_config['directories']['base'] = dataset_dir
    
    # Save project config
    with open(config_file, 'w') as f:
        yaml.dump(project_config, f, default_flow_style=False, indent=2)
    
    print(f"\n✅ Project created successfully!")
    print(f"📄 Config file: {config_file}")
    print(f"📁 Dataset directory: {dataset_dir}")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Copy your config: cp {config_file} project_config.yaml")
    print(f"   2. Create directories: mkdir -p raw_videos/New raw_videos/Processed")
    print(f"   3. Add videos to: raw_videos/New/")
    print(f"   4. Run extraction: python image_extraction.py")
    print(f"   5. Start annotation: python annotation_tool.py")
    
    return config_file

def list_available_projects():
    """List all available project configs"""
    
    config_files = list(Path('.').glob('*_config.yaml'))
    
    if not config_files:
        print("📁 No project configs found")
        return
    
    print("📋 Available Projects:")
    print("-" * 30)
    
    for i, config_file in enumerate(config_files, 1):
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            name = config_data.get('project', {}).get('name', 'Unknown')
            description = config_data.get('project', {}).get('description', 'No description')
            classes = config_data.get('dataset', {}).get('classes', [])
            
            print(f"   {i}. {name}")
            print(f"      File: {config_file}")
            print(f"      Description: {description}")
            print(f"      Classes: {', '.join(classes)} ({len(classes)} total)")
            print()
            
        except Exception as e:
            print(f"   {i}. {config_file} (Error reading: {e})")

def switch_project():
    """Switch to a different project config"""
    
    print("🔄 Project Switcher")
    print("=" * 30)
    
    list_available_projects()
    
    config_files = list(Path('.').glob('*_config.yaml'))
    if not config_files:
        return
    
    try:
        choice = int(input(f"\nSelect project (1-{len(config_files)}): ")) - 1
        if 0 <= choice < len(config_files):
            selected_file = config_files[choice]
            
            # Copy selected config to active config
            import shutil
            shutil.copy(selected_file, 'project_config.yaml')
            
            print(f"✅ Switched to project: {selected_file}")
            print("🔄 Restart your tools to use the new project config")
        else:
            print("❌ Invalid selection")
    except ValueError:
        print("❌ Invalid input")

if __name__ == "__main__":
    print("🎯 Project Management System")
    print("=" * 40)
    print("1. Create new project")
    print("2. List available projects") 
    print("3. Switch project")
    print("4. Exit")
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == '1':
        create_new_project()
    elif choice == '2':
        list_available_projects()
    elif choice == '3':
        switch_project()
    elif choice == '4':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")