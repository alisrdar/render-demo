from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

MODEL_PATH = "best_priority_model"

# Load model and tokenizer once when app starts
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    # You might want to handle this more gracefully in production

label_map = {
    0: "Low",
    1: "Medium",
    2: "High"
}

def predict_priority(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

    return label_map[predicted_class.item()], round(confidence.item() * 100, 2)

# --- Original Route (Kept for the web interface) ---
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    complaint_text = ""

    if request.method == "POST":
        complaint_text = request.form.get("complaint", "")
        if complaint_text:
            prediction, confidence = predict_priority(complaint_text)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        complaint_text=complaint_text
    )

# --- NEW API Route (For your Mobile App) ---
@app.route("/api/predict", methods=["POST"])
def api_predict():
    # 1. Get JSON data from the request
    data = request.get_json(force=True, silent=True)
    
    if not data or 'complaint' not in data:
        return jsonify({'error': 'No complaint text provided'}), 400

    complaint_text = data['complaint']

    # 2. Run prediction
    try:
        prediction, confidence = predict_priority(complaint_text)
        
        # 3. Return JSON response
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)