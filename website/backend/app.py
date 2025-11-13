import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS  # ADD THIS
import io

# ------------------------------
# 1. Ρύθμιση Flask app
# ------------------------------
app = Flask(__name__)
CORS(app)  # ADD THIS

# ------------------------------
# 2. Φόρτωση του μοντέλου
# ------------------------------
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("🚀 Loading model...")
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load("brisc2025_cnn_final.pth", map_location=device))
model.to(device)
model.eval()
print("✅ Model loaded successfully!")

# ------------------------------
# 3. Μετασχηματισμός εικόνας
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------
# 4. Endpoint για πρόβλεψη
# ------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'})

    file = request.files['file']
    if not file:
        return jsonify({'error': 'empty file'})

    try:
        # Μετατροπή σε PIL image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Μετασχηματισμός
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Πρόβλεψη
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top_prob, top_class = torch.max(probs, dim=0)

        # Αν ο top_class != 'no_tumor', θεωρούμε ύπαρξη καρκίνου
        result = {
            "predicted_class": class_names[top_class.item()],
            "confidence": float(top_prob.item() * 100),
            "has_tumor": class_names[top_class.item()] != "no_tumor"
        }

        print(f"📊 Prediction: {result['predicted_class']} (confidence: {result['confidence']:.2f}%)")
        return jsonify(result)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)})

# Health check endpoint for testing
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Server is running'})

# ------------------------------
# 5. Εκκίνηση
# ------------------------------
if __name__ == '__main__':
    print("✅ Server starting on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)