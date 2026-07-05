#!/usr/bin/env python3
"""
Plastic Waste Classifier - Terminal Runner
Run trained ResNet18 model for plastic classification
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys
import argparse

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Plastic information database
plastic_db = {
    'PET': {
        'full_name': 'Polyethylene Terephthalate',
        'recycling_code': '#1 PET',
        'recyclability': 'Highly Recyclable',
        'color': 'green',
        'common_uses': 'Water bottles, soft drink bottles, food containers',
        'recycling_tips': 'Rinse and remove caps before recycling'
    },
    'HDPE': {
        'full_name': 'High-Density Polyethylene',
        'recycling_code': '#2 HDPE',
        'recyclability': 'Highly Recyclable',
        'color': 'green',
        'common_uses': 'Milk jugs, detergent bottles, shampoo bottles',
        'recycling_tips': 'Clean and dry before recycling'
    },
    'PVC': {
        'full_name': 'Polyvinyl Chloride',
        'recycling_code': '#3 PVC',
        'recyclability': 'Rarely Recyclable',
        'color': 'red',
        'common_uses': 'Pipes, vinyl siding, credit cards',
        'recycling_tips': 'Check with local facilities, often not accepted'
    },
    'LDPE': {
        'full_name': 'Low-Density Polyethylene',
        'recycling_code': '#4 LDPE',
        'recyclability': 'Sometimes Recyclable',
        'color': 'orange',
        'common_uses': 'Plastic bags, squeeze bottles, bread bags',
        'recycling_tips': 'Many stores accept plastic bags for recycling'
    },
    'PP': {
        'full_name': 'Polypropylene',
        'recycling_code': '#5 PP',
        'recyclability': 'Recyclable',
        'color': 'blue',
        'common_uses': 'Yogurt containers, bottle caps, straws',
        'recycling_tips': 'Clean containers before recycling'
    },
    'PS': {
        'full_name': 'Polystyrene',
        'recycling_code': '#6 PS',
        'recyclability': 'Rarely Recyclable',
        'color': 'red',
        'common_uses': 'Foam cups, takeout containers, packing peanuts',
        'recycling_tips': 'Most facilities do not accept, check locally'
    },
    'OTHERS': {
        'full_name': 'Other Plastics',
        'recycling_code': '#7 OTHER',
        'recyclability': 'Check Guidelines',
        'color': 'gray',
        'common_uses': 'Mixed plastics, polycarbonate, bioplastics',
        'recycling_tips': 'Varies by type, check with local recycling center'
    }
}

class PlasticClassifier:
    def __init__(self, model_path='best_model_ResNet18.pth'):
        """Initialize the classifier with trained model"""
        self.model_path = model_path
        self.device = device
        self.model = None
        self.idx_to_class = None
        self.num_classes = None
        self.best_acc = None
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"🔄 Loading model from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.num_classes = checkpoint['num_classes']
        self.idx_to_class = checkpoint['idx_to_class']
        self.best_acc = checkpoint['best_acc']
        
        # Create model architecture
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"🎯 Classes: {list(self.idx_to_class.values())}")
        print(f"🏆 Best Accuracy: {self.best_acc:.4f} ({self.best_acc*100:.2f}%)")
        print(f"🖥️  Device: {self.device}\n")
    
    def predict(self, image_path):
        """Predict plastic type from image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
            
            # Convert idx_to_class keys to integers if they're strings
            if isinstance(list(self.idx_to_class.keys())[0], str):
                idx_to_class_int = {int(k): v for k, v in self.idx_to_class.items()}
            else:
                idx_to_class_int = self.idx_to_class
            
            predicted_idx = predicted.item()
            plastic_type = idx_to_class_int.get(predicted_idx, 'UNKNOWN')
            confidence_score = confidence.item()
            
            # Get all class probabilities
            all_probs = probs[0].cpu().numpy()
            
            return {
                'success': True,
                'plastic_type': plastic_type,
                'confidence': confidence_score,
                'info': plastic_db.get(plastic_type, {}),
                'all_probabilities': {idx_to_class_int[i]: float(all_probs[i]) for i in range(len(all_probs))}
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def display_results(self, result, image_path):
        """Display prediction results"""
        print('\n' + '='*70)
        print('🔍 PLASTIC CLASSIFICATION RESULTS')
        print('='*70)
        print(f"📁 Image: {os.path.basename(image_path)}")
        
        if not result['success']:
            print(f"\n❌ Error: {result['error']}")
            print('='*70)
            return
        
        plastic_type = result['plastic_type']
        confidence = result['confidence']
        info = result['info']
        
        # Main prediction
        print(f"\n🎯 Predicted Type: {plastic_type}")
        print(f"📊 Confidence: {confidence*100:.2f}%")
        
        if info:
            print(f"\n📝 Full Name: {info['full_name']}")
            print(f"♻️  Recycling Code: {info['recycling_code']}")
            print(f"🌍 Recyclability: {info['recyclability']}")
            print(f"📦 Common Uses: {info['common_uses']}")
            print(f"💡 Recycling Tips: {info['recycling_tips']}")
        
        # All probabilities
        print(f"\n📈 All Class Probabilities:")
        for cls, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
            bar = '█' * int(prob * 50)
            print(f"  {cls:8s}: {prob*100:5.2f}% {bar}")
        
        print('='*70 + '\n')
    
    def batch_predict(self, folder_path, max_images=None):
        """Predict on multiple images in a folder"""
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return
        
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print("❌ No images found in folder")
            return
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"\n📁 Processing {len(image_files)} images from {folder_path}...\n")
        
        results_summary = {}
        
        for i, img_file in enumerate(image_files, 1):
            img_path = os.path.join(folder_path, img_file)
            result = self.predict(img_path)
            
            if result['success']:
                plastic_type = result['plastic_type']
                confidence = result['confidence']
                print(f"{i:3d}. {img_file:30s} → {plastic_type:8s} ({confidence*100:.1f}%)")
                
                if plastic_type not in results_summary:
                    results_summary[plastic_type] = 0
                results_summary[plastic_type] += 1
        
        # Summary
        print("\n" + "="*70)
        print("📊 BATCH PREDICTION SUMMARY")
        print("="*70)
        for plastic_type, count in sorted(results_summary.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(image_files)) * 100
            print(f"  {plastic_type:8s}: {count:3d} images ({percentage:5.1f}%)")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plastic Waste Classifier - Terminal Runner')
    parser.add_argument('--image', '-i', type=str, help='Path to single image file')
    parser.add_argument('--folder', '-f', type=str, help='Path to folder with multiple images')
    parser.add_argument('--max', '-m', type=int, default=None, help='Maximum number of images to process')
    parser.add_argument('--model', type=str, default='best_model_ResNet18.pth', help='Path to model file')
    
    args = parser.parse_args()
    
    # Print header
    print('\n' + '='*70)
    print('♻️  PLASTIC WASTE CLASSIFIER - TERMINAL RUNNER')
    print('='*70 + '\n')
    
    # Initialize classifier
    try:
        classifier = PlasticClassifier(model_path=args.model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Process based on arguments
    if args.image:
        # Single image prediction
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            return
        
        result = classifier.predict(args.image)
        classifier.display_results(result, args.image)
    
    elif args.folder:
        # Batch prediction
        classifier.batch_predict(args.folder, max_images=args.max)
    
    else:
        # No arguments - show usage
        print("Usage:")
        print("  Single image:  python run_classifier.py --image <path_to_image>")
        print("  Batch folder:  python run_classifier.py --folder <path_to_folder>")
        print("  Limit images:  python run_classifier.py --folder <path> --max 10")
        print("\nExamples:")
        print('  python run_classifier.py --image "test/PET_1.jpg"')
        print('  python run_classifier.py --folder "test" --max 5')
        print("\nFor web interface:")
        print("  python app.py")
        print("  Then open: http://localhost:5000")
        print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
