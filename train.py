from ultralytics import YOLO
import os
import yaml

def create_dataset_yaml():
    """Create dataset configuration file"""
    dataset_config = {
        'path': './dataset',       # Dataset root path
        'train': 'train/images',   # Train images (relative to path)
        'val': 'val/images',     # Val images (relative to path) ⚠️ pakai valid
        'test': 'test/images',     # Test images (optional)
        'nc': 1,                   # Number of classes
        'names': ['monyet']        # Class names
    }
    
    os.makedirs('dataset', exist_ok=True)
    
    with open('dataset/data.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print("✅ Dataset configuration created: dataset/data.yaml")

def train_model():
    """Train YOLOv8 model for monkey detection"""
    
    if not os.path.exists('dataset/data.yaml'):
        create_dataset_yaml()
    
    if not os.path.exists('dataset/train/images') or not os.path.exists('dataset/val/images'):
        print("❌ Error: Dataset not found!")
        print("Please ensure the following structure:")
        print("dataset/")
        print("  train/")
        print("    images/  # Training images (.jpg, .png)")
        print("    labels/  # YOLO format labels (.txt)")
        print("  val/")
        print("    images/  # Validation images")
        print("    labels/  # Validation labels")
        return
    
    try:
        model = YOLO('yolov8n.pt')  # Pretrained model
        print("🚀 Starting training...")
        results = model.train(
            data='dataset/data.yaml',
            epochs=100,
            imgsz=640,
            batch=16,
            name='monyet',          # Folder hasil training
            patience=10,
            save=True,
            plots=True,
            device='cpu',             # Gunakan GPU 0, atau 'cpu' jika tidak ada GPU
            workers=4,
            project='runs/detect'
        )
        print("✅ Training completed!")
        print(f"Best model saved at: runs/detect/monyet/weights/best.pt")
        
        print("\n🔎 Validating model...")
        metrics = model.val()
        print(f"Validation results: {metrics}")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("Ensure proper dataset structure, YOLO format labels, and sufficient hardware resources.")

def export_model():
    """Export trained model to different formats"""
    try:
        model_path = 'runs/detect/monyet/weights/best.pt'
        if not os.path.exists(model_path):
            print("❌ No trained model found. Please train the model first.")
            return
        
        model = YOLO(model_path)
        print("📤 Exporting model...")
        model.export(format='onnx')
        print("✅ Model exported to ONNX format")
        
    except Exception as e:
        print(f"❌ Export error: {e}")

if __name__ == '__main__':
    print("YOLOv8 Monkey Detection Training Script")
    print("=" * 50)
    
    choice = input("Choose an option:\n1. Create dataset config\n2. Train model\n3. Export model\nEnter choice (1-3): ")
    
    if choice == '1':
        create_dataset_yaml()
    elif choice == '2':
        train_model()
    elif choice == '3':
        export_model()
    else:
        print("Invalid choice. Please run the script again.")
