"""
=============================================================================
Railway Track Defect Detection - Flask Backend (app.py)
=============================================================================
Academic Neural Network Mini-Project

Flask Web Server serving trained CNN & MobileNetV2 models.
Provides real-time image upload, risk level evaluation, PDF report generation parameters,
sample gallery listing, and prediction endpoints.
=============================================================================
"""

import os
import json
import time
import random
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for

# Optional TensorFlow import with dynamic error handling if model/TF is pending
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'class_names.json')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_FILE_SIZE_MB = 10

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024  # 10 MB limit

# Global variables for loaded models and class mapping
CNN_MODEL = None
MOBILENET_MODEL = None
CLASS_MAPPING = {"0": "Defective", "1": "Non-defective"}  # Fallback default

def load_models():
    """Loads trained .keras models and class names mapping file at startup."""
    global CNN_MODEL, MOBILENET_MODEL, CLASS_MAPPING
    
    # Load class mapping
    if os.path.exists(CLASS_NAMES_PATH):
        try:
            with open(CLASS_NAMES_PATH, 'r') as f:
                CLASS_MAPPING = json.load(f)
            print(f"--> Loaded Class Mapping from JSON: {CLASS_MAPPING}")
        except Exception as e:
            print(f"[WARN] Error loading class_names.json: {e}")
            
    # Load CNN Model from Scratch
    cnn_path = os.path.join(MODELS_DIR, 'railway_track_cnn.keras')
    if os.path.exists(cnn_path) and TF_AVAILABLE:
        try:
            CNN_MODEL = tf.keras.models.load_model(cnn_path)
            print(f"--> Successfully loaded CNN Scratch Model from {cnn_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load CNN model: {e}")
            CNN_MODEL = None

    # Load MobileNetV2 Model
    mb_path = os.path.join(MODELS_DIR, 'railway_track_mobilenet.keras')
    if os.path.exists(mb_path) and TF_AVAILABLE:
        try:
            MOBILENET_MODEL = tf.keras.models.load_model(mb_path)
            print(f"--> Successfully loaded MobileNetV2 Model from {mb_path}")
        except Exception as e:
            print(f"[WARN] Failed to load MobileNetV2 model: {e}")
            MOBILENET_MODEL = None

load_models()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(target_size)
    img_array = np.array(img_resized, dtype=np.float32)  # Keep in [0, 255] range for model Rescaling / preprocess_input
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch, img.size

def calculate_risk_and_recommendation(is_defective, confidence_score):
    """
    Risk & Severity Assessment Engine:
    Calculates Severity Score (0.0 - 5.0 scale), Risk Level, and Official Maintenance Action.
    """
    if is_defective:
        if confidence_score >= 80.0:
            risk_level = "CRITICAL RISK"
            risk_code = "CRITICAL"
            severity_score = round(3.5 + ((confidence_score - 80.0) / 20.0) * 1.5, 1)
            status_title = "CRITICAL SURFACE FRACTURE DETECTED"
            recommendation = "CRITICAL DANGER: Immediate track section closure required. Dispatch emergency track maintenance crew for manual inspection, rail section replacement, and structural fastener reinforcement."
        else:
            risk_level = "MODERATE RISK"
            risk_code = "MODERATE"
            severity_score = round(2.0 + (confidence_score / 80.0) * 1.5, 1)
            status_title = "SURFACE DEFECT DETECTED"
            recommendation = "PRIORITY WARNING: Flag track section for priority inspection within 24-48 hours. Monitor defect progression and schedule speed restriction."
    else:
        risk_level = "LOW RISK"
        risk_code = "LOW"
        severity_score = round((100.0 - confidence_score) / 100.0 * 1.0, 1)
        status_title = "TRACK OPERATIONAL & CLEAR"
        recommendation = "TRACK CLEAR: No immediate maintenance required. Maintain routine periodic visual & ultrasonic track inspection schedule."

    return {
        "risk_level": risk_level,
        "risk_code": risk_code,
        "severity_score": severity_score,
        "status_title": status_title,
        "recommendation": recommendation
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/results/<path:filename>')
def serve_result_image(filename):
    return send_from_directory(RESULTS_DIR, filename)

@app.route('/api/metrics')
def get_metrics():
    cnn_metrics = None
    mobilenet_metrics = None
    
    cnn_metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    if os.path.exists(cnn_metrics_path):
        with open(cnn_metrics_path, 'r') as f:
            cnn_metrics = json.load(f)
            
    mobilenet_metrics_path = os.path.join(RESULTS_DIR, 'metrics_mobilenet.json')
    if os.path.exists(mobilenet_metrics_path):
        with open(mobilenet_metrics_path, 'r') as f:
            mobilenet_metrics = json.load(f)
            
    return jsonify({
        "status": "success",
        "cnn": cnn_metrics,
        "mobilenet": mobilenet_metrics,
        "model_loaded": (CNN_MODEL is not None)
    })

@app.route('/api/samples')
def get_samples():
    """Returns sample test images for 1-click test gallery."""
    samples = []
    test_dir = os.path.join(DATASET_DIR, 'test')
    
    if os.path.exists(test_dir):
        for category in ['Defective', 'Non-defective']:
            cat_path = os.path.join(test_dir, category)
            if os.path.exists(cat_path):
                files = [f for f in os.listdir(cat_path) if allowed_file(f)]
                selected = files[:3]  # Take first 3 sample images per class
                for f in selected:
                    rel_path = f"dataset/test/{category}/{f}"
                    samples.append({
                        "name": f"{category} Sample",
                        "category": category,
                        "filename": f,
                        "url": f"/{rel_path}"
                    })

    return jsonify({"samples": samples})

@app.route('/dataset/test/<path:filename>')
def serve_sample_image(filename):
    return send_from_directory(os.path.join(DATASET_DIR, 'test'), filename)

@app.route('/predict', methods=['POST'])
def predict():
    global CNN_MODEL, MOBILENET_MODEL
    
    if CNN_MODEL is None and TF_AVAILABLE:
        load_models()
        
    model_type = request.form.get('model_type', 'cnn').lower()
    selected_model = MOBILENET_MODEL if (model_type == 'mobilenet' and MOBILENET_MODEL is not None) else CNN_MODEL
    model_name_display = "MobileNetV2 Transfer Learning" if (model_type == 'mobilenet' and MOBILENET_MODEL is not None) else "CNN From Scratch (4 Blocks)"
    
    if selected_model is None:
        return jsonify({
            'error': 'Trained model file is not loaded or missing. Please run python train.py first.'
        }), 400

    saved_filename = None
    filepath = None
    original_filename = "sample.jpg"

    # Support file upload or sample image selection
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid format. Allowed: {", ".join(ALLOWED_EXTENSIONS).upper()}'}), 400
        original_filename = secure_filename(file.filename)
        saved_filename = f"{int(time.time())}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)

    elif request.form.get('sample_url'):
        sample_rel_path = request.form.get('sample_url').lstrip('/')
        sample_full_path = os.path.join(BASE_DIR, sample_rel_path.replace('/', os.sep))
        if os.path.exists(sample_full_path):
            original_filename = os.path.basename(sample_full_path)
            saved_filename = f"{int(time.time())}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
            img = Image.open(sample_full_path)
            img.save(filepath)
        else:
            return jsonify({'error': 'Selected sample image file not found.'}), 400
    else:
        return jsonify({'error': 'No image file or sample provided.'}), 400

    try:
        start_time = time.time()
        img_batch, orig_dimensions = preprocess_image(filepath)
        
        # Model Prediction
        prediction_prob = float(selected_model.predict(img_batch, verbose=0)[0][0])
        
        class_0_label = CLASS_MAPPING.get("0", "Defective")
        class_1_label = CLASS_MAPPING.get("1", "Non-defective")
        
        prob_index_1 = prediction_prob
        prob_index_0 = 1.0 - prediction_prob
        
        if prob_index_1 >= 0.5:
            predicted_class_index = 1
            raw_class_name = class_1_label
            confidence_score = prob_index_1 * 100.0
        else:
            predicted_class_index = 0
            raw_class_name = class_0_label
            confidence_score = prob_index_0 * 100.0
            
        if "non-defective" in class_1_label.lower() or "normal" in class_1_label.lower():
            normal_prob = prob_index_1 * 100.0
            defective_prob = prob_index_0 * 100.0
        else:
            defective_prob = prob_index_1 * 100.0
            normal_prob = prob_index_0 * 100.0
            
        is_defective = ("defective" in raw_class_name.lower()) and ("non" not in raw_class_name.lower())
        human_readable = "DEFECTIVE TRACK" if is_defective else "NORMAL TRACK"
        
        # Risk & Severity Assessment
        risk_info = calculate_risk_and_recommendation(is_defective, confidence_score)
        
        process_time = round((time.time() - start_time) * 1000, 2)
        image_url = url_for('static', filename=f'uploads/{saved_filename}')
        report_id = f"RG-{time.strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
        
        return jsonify({
            'success': True,
            'prediction': human_readable,
            'raw_class': raw_class_name,
            'is_defective': is_defective,
            'confidence': round(confidence_score, 2),
            'normal_probability': round(normal_prob, 2),
            'defective_probability': round(defective_prob, 2),
            'image_url': image_url,
            'filename': original_filename,
            'dimensions': f"{orig_dimensions[0]} x {orig_dimensions[1]}",
            'process_time_ms': process_time,
            'model_name': model_name_display,
            'report_id': report_id,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'risk': risk_info
        })
        
    except Exception as e:
        print(f"[ERROR] Exception during prediction: {e}")
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': f'File exceeds maximum size limit of {MAX_FILE_SIZE_MB} MB.'}), 413

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

if __name__ == '__main__':
    print("="*60)
    print("Starting RailGuard AI - Flask Web Server")
    print("Server running at: http://127.0.0.1:5000/")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=True)
