from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('best_model_ResNet18.pth', map_location=device)
num_classes = checkpoint['num_classes']
idx_to_class = checkpoint['idx_to_class']

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

plastic_db = {"PET": {"full_name": "Polyethylene Terephthalate", "recycling_code": "#1 PET", "recyclability": "Highly Recyclable", "color": "green"}, "HDPE": {"full_name": "High-Density Polyethylene", "recycling_code": "#2 HDPE", "recyclability": "Highly Recyclable", "color": "green"}, "PVC": {"full_name": "Polyvinyl Chloride", "recycling_code": "#3 PVC", "recyclability": "Rarely Recyclable", "color": "red"}, "LDPE": {"full_name": "Low-Density Polyethylene", "recycling_code": "#4 LDPE", "recyclability": "Sometimes Recyclable", "color": "orange"}, "PP": {"full_name": "Polypropylene", "recycling_code": "#5 PP", "recyclability": "Recyclable", "color": "blue"}, "PS": {"full_name": "Polystyrene", "recycling_code": "#6 PS", "recyclability": "Rarely Recyclable", "color": "red"}, "OTHERS": {"full_name": "Other Plastics", "recycling_code": "#7 OTHER", "recyclability": "Check Guidelines", "color": "gray"}}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/classify', methods=['POST'])
def classify():
    try:
        file = request.files['file']
        image = Image.open(file.stream).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        # Convert idx_to_class keys to integers if they're strings
        if isinstance(list(idx_to_class.keys())[0], str):
            idx_to_class_int = {int(k): v for k, v in idx_to_class.items()}
        else:
            idx_to_class_int = idx_to_class

        predicted_idx = predicted.item()
        plastic_type = idx_to_class_int.get(predicted_idx, 'UNKNOWN')

        return jsonify({
            'plastic_type': plastic_type,
            'confidence': float(confidence.item()),
            'info': plastic_db.get(plastic_type, {}),
            'success': True
        })
    except Exception as e:
        print(f"Error in classify: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
