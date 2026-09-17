"""
=============================================================================
Railway Track Defect Detection - CNN Training Script (train.py)
=============================================================================
Academic Neural Network Mini-Project

This script performs end-to-end training of a Convolutional Neural Network (CNN)
to classify railway track images into DEFECTIVE or NON-DEFECTIVE (NORMAL).

Key Stages:
1. Dataset Loading & Class Mapping detection
2. Data Augmentation & Normalization (Scaling pixel values [0, 255] to [0, 1])
3. CNN Model Construction from Scratch (Conv2D, ReLU, MaxPool, Flatten, Dense, Dropout, Sigmoid)
4. Model Compilation with Adam Optimizer & Binary Crossentropy Loss
5. Model Training with EarlyStopping & ReduceLROnPlateau Callbacks
6. Model Evaluation on Test Set (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)
7. Exporting trained model (.keras), class mapping (JSON), and evaluation visualizations (PNG)
=============================================================================
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def plot_training_history(history):
    """Plots and saves Accuracy and Loss curves using pure Pillow (bypassing GUI/DLL dependencies)."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = len(acc)
    
    # 1. Plot Accuracy Chart
    img_acc = Image.new('RGB', (800, 500), color=(28, 37, 65))
    draw_acc = ImageDraw.Draw(img_acc)
    
    # Title & Axis Labels
    draw_acc.text((250, 20), "CNN Model Training & Validation Accuracy", fill=(248, 250, 252))
    draw_acc.text((370, 460), "Epochs", fill=(148, 163, 184))
    draw_acc.text((20, 240), "Accuracy", fill=(148, 163, 184))
    
    # Draw Graph Box (X: 80 -> 750, Y: 60 -> 440)
    draw_acc.rectangle([80, 60, 750, 440], outline=(148, 163, 184), width=2)
    
    # Grid lines & ticks
    for i in range(5):
        y_val = 60 + i * 95
        draw_acc.line([(80, y_val), (750, y_val)], fill=(45, 55, 85), width=1)
        acc_label = f"{1.0 - i*0.2:.1f}"
        draw_acc.text((40, y_val - 6), acc_label, fill=(148, 163, 184))
        
    # Plot accuracy data points
    dx = (750 - 80) / max(epochs - 1, 1)
    acc_pts, val_acc_pts = [], []
    for i in range(epochs):
        x = 80 + i * dx
        y_a = 440 - acc[i] * 380
        y_va = 440 - val_acc[i] * 380
        acc_pts.append((x, y_a))
        val_acc_pts.append((x, y_va))
        
    for i in range(len(acc_pts) - 1):
        draw_acc.line([acc_pts[i], acc_pts[i+1]], fill=(58, 134, 255), width=3)
        draw_acc.line([val_acc_pts[i], val_acc_pts[i+1]], fill=(6, 214, 160), width=3)
        
    for pt in acc_pts:
        draw_acc.ellipse([pt[0]-4, pt[1]-4, pt[0]+4, pt[1]+4], fill=(58, 134, 255))
    for pt in val_acc_pts:
        draw_acc.ellipse([pt[0]-4, pt[1]-4, pt[0]+4, pt[1]+4], fill=(6, 214, 160))
        
    # Legend
    draw_acc.rectangle([550, 80, 730, 140], fill=(11, 19, 43), outline=(148, 163, 184))
    draw_acc.line([(560, 100), (590, 100)], fill=(58, 134, 255), width=3)
    draw_acc.text((600, 93), "Train Acc", fill=(248, 250, 252))
    draw_acc.line([(560, 120), (590, 120)], fill=(6, 214, 160), width=3)
    draw_acc.text((600, 113), "Val Acc", fill=(248, 250, 252))
    
    acc_plot_path = os.path.join(RESULTS_DIR, 'accuracy.png')
    img_acc.save(acc_plot_path)
    print(f"--> Saved Accuracy Plot to {acc_plot_path}")
    
    # 2. Plot Loss Chart
    img_loss = Image.new('RGB', (800, 500), color=(28, 37, 65))
    draw_loss = ImageDraw.Draw(img_loss)
    
    draw_loss.text((260, 20), "CNN Model Training & Validation Loss", fill=(248, 250, 252))
    draw_loss.text((370, 460), "Epochs", fill=(148, 163, 184))
    draw_loss.text((20, 240), "Loss", fill=(148, 163, 184))
    draw_loss.rectangle([80, 60, 750, 440], outline=(148, 163, 184), width=2)
    
    max_loss = max(max(loss), max(val_loss), 1.0)
    for i in range(5):
        y_val = 60 + i * 95
        draw_loss.line([(80, y_val), (750, y_val)], fill=(45, 55, 85), width=1)
        l_label = f"{max_loss * (1.0 - i*0.25):.2f}"
        draw_loss.text((35, y_val - 6), l_label, fill=(148, 163, 184))
        
    loss_pts, val_loss_pts = [], []
    for i in range(epochs):
        x = 80 + i * dx
        y_l = 440 - (loss[i] / max_loss) * 380
        y_vl = 440 - (val_loss[i] / max_loss) * 380
        loss_pts.append((x, y_l))
        val_loss_pts.append((x, y_vl))
        
    for i in range(len(loss_pts) - 1):
        draw_loss.line([loss_pts[i], loss_pts[i+1]], fill=(230, 57, 70), width=3)
        draw_loss.line([val_loss_pts[i], val_loss_pts[i+1]], fill=(255, 159, 28), width=3)
        
    for pt in loss_pts:
        draw_loss.ellipse([pt[0]-4, pt[1]-4, pt[0]+4, pt[1]+4], fill=(230, 57, 70))
    for pt in val_loss_pts:
        draw_loss.ellipse([pt[0]-4, pt[1]-4, pt[0]+4, pt[1]+4], fill=(255, 159, 28))
        
    draw_loss.rectangle([550, 80, 730, 140], fill=(11, 19, 43), outline=(148, 163, 184))
    draw_loss.line([(560, 100), (590, 100)], fill=(230, 57, 70), width=3)
    draw_loss.text((600, 93), "Train Loss", fill=(248, 250, 252))
    draw_loss.line([(560, 120), (590, 120)], fill=(255, 159, 28), width=3)
    draw_loss.text((600, 113), "Val Loss", fill=(248, 250, 252))
    
    loss_plot_path = os.path.join(RESULTS_DIR, 'loss.png')
    img_loss.save(loss_plot_path)
    print(f"--> Saved Loss Plot to {loss_plot_path}")

def plot_confusion_matrix(cm, class_names):
    """Plots and saves the Confusion Matrix heatmap using pure Pillow."""
    img_cm = Image.new('RGB', (600, 500), color=(28, 37, 65))
    draw = ImageDraw.Draw(img_cm)
    
    draw.text((180, 25), "Railway Track CNN - Confusion Matrix", fill=(248, 250, 252))
    draw.text((250, 455), "Predicted Class", fill=(148, 163, 184))
    draw.text((15, 230), "Actual", fill=(148, 163, 184))
    
    # 2x2 Grid Boxes (Box size: 160x160)
    grid_start_x, grid_start_y = 140, 90
    box_w, box_h = 170, 150
    
    max_count = max(np.max(cm), 1)
    
    for r in range(2):
        for c in range(2):
            count = cm[r][c]
            x1 = grid_start_x + c * box_w
            y1 = grid_start_y + r * box_h
            x2 = x1 + box_w
            y2 = y1 + box_h
            
            # Heatmap intensity
            intensity = int(50 + (count / max_count) * 180)
            color = (30, intensity, int(intensity * 0.8))
            
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(148, 163, 184), width=2)
            draw.text((x1 + box_w//2 - 10, y1 + box_h//2 - 10), str(count), fill=(255, 255, 255))
            
    # Class Axis Labels
    c0 = class_names[0] if len(class_names) > 0 else "0"
    c1 = class_names[1] if len(class_names) > 1 else "1"
    
    draw.text((grid_start_x + 40, grid_start_y - 30), c0, fill=(248, 250, 252))
    draw.text((grid_start_x + box_w + 40, grid_start_y - 30), c1, fill=(248, 250, 252))
    
    draw.text((grid_start_x - 90, grid_start_y + 60), c0, fill=(248, 250, 252))
    draw.text((grid_start_x - 90, grid_start_y + box_h + 60), c1, fill=(248, 250, 252))
    
    cm_plot_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    img_cm.save(cm_plot_path)
    print(f"--> Saved Confusion Matrix to {cm_plot_path}")

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Pure NumPy implementation of classification evaluation metrics
def compute_binary_metrics(y_true, y_pred):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    cm = np.array([[tn, fp], [fn, tp]])
    return accuracy, precision, recall, f1, cm

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Directory configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'validation')
TEST_DIR = os.path.join(DATASET_DIR, 'test')

MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
EPOCHS = 25

def check_dataset_exists():
    """Validates that the dataset directories exist before proceeding."""
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print("\n[ERROR] Dataset directory not found!")
        print("Please run 'python create_sample_dataset.py' to generate a sample dataset,")
        print("or place your custom railway track images into the dataset/ directory.")
        exit(1)

def load_datasets():
    """
    Loads dataset splits using tf.keras.utils.image_dataset_from_directory.
    Returns training, validation, and test datasets along with detected class names.
    """
    print("Loading Railway Track Dataset...")
    
    # Load Training Set
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    # Load Validation Set
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Load Test Set
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    class_names = train_ds.class_names
    print(f"--> Discovered Classes: {class_names}")
    
    # Save exact class mapping to JSON to prevent class label reversal during prediction
    # TensorFlow assigns integer index according to alphabetical order of folder names
    class_mapping = {str(i): name for i, name in enumerate(class_names)}
    mapping_path = os.path.join(BASE_DIR, 'class_names.json')
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=4)
    print(f"--> Preserved Class Mapping saved to {mapping_path}: {class_mapping}")
    
    return train_ds, val_ds, test_ds, class_names

def build_data_augmentation():
    """
    Builds data augmentation pipeline for training dataset.
    Augmentation improves model generalization and prevents overfitting by exposing
    the CNN to varied rotations, flips, zoom levels, and contrast shifts.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1)
    ], name="data_augmentation")

def build_cnn_model():
    """
    Constructs a 4-Block Convolutional Neural Network (CNN) architecture from scratch.
    
    Architecture Explanation for Viva:
    1. Rescaling (1./255): Normalizes pixel values from [0, 255] RGB range to [0, 1] range for stable gradient descent.
    2. Conv2D Layers: Applies learnable 3x3 filters to extract local visual patterns (edges, textures, rail breaks).
    3. ReLU Activation: Adds non-linearity f(x) = max(0, x), allowing the network to learn complex patterns.
    4. MaxPooling2D (2x2): Reduces spatial dimensions by keeping maximum values, reducing parameters & computation.
    5. Flatten: Converts 2D feature maps into a 1D feature vector for classification.
    6. Dense Layer (128 units): High-level feature representation layer.
    7. Dropout (0.5): Randomly deactivates 50% of neurons during training to mitigate overfitting.
    8. Dense Output (1 unit, Sigmoid): Produces probability score between 0.0 and 1.0.
    """
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Data Augmentation & Normalization
    x = build_data_augmentation()(inputs)
    x = layers.Rescaling(1./255)(x)
    
    # Block 1: 32 Filters
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 2: 64 Filters
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 3: 128 Filters
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 4: 256 Filters
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Classification Head
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Railway_Track_CNN")
    return model



def train_and_evaluate():
    check_dataset_exists()
    train_ds, val_ds, test_ds, class_names = load_datasets()
    
    # Build CNN Architecture
    model = build_cnn_model()
    print("\n" + "="*60)
    print("CNN MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    model.summary()
    
    # Compile Model with tuned Adam optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    model_save_path = os.path.join(MODELS_DIR, 'railway_track_cnn.keras')
    checkpoint = callbacks.ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
    
    print("\nStarting CNN Model Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    # Plot accuracy and loss curves
    plot_training_history(history)
    
    # Load best model for evaluation
    print("\nLoading best trained model for evaluation on test dataset...")
    best_model = models.load_model(model_save_path)
    
    # Evaluate on Test Set
    y_true = []
    y_pred_probs = []
    
    for images, labels in test_ds:
        probs = best_model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(probs.flatten())
        
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs >= 0.5).astype(int)
    
    # Calculate Metrics using pure numpy implementation
    acc, prec, rec, f1, cm = compute_binary_metrics(y_true, y_pred)
    
    print("\n" + "="*60)
    print("EVALUATION METRICS ON TEST SET")
    print("="*60)
    print(f"Test Accuracy  : {acc * 100:.2f}%")
    print(f"Precision      : {prec * 100:.2f}%")
    print(f"Recall         : {rec * 100:.2f}%")
    print(f"F1-Score       : {f1 * 100:.2f}%")
    print("\nConfusion Matrix:\n", cm)
    
    # Plot Confusion Matrix
    plot_confusion_matrix(cm, class_names)
    
    # Save Metrics JSON
    metrics_data = {
        "model_name": "Convolutional Neural Network (CNN) Scratch",
        "test_accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "epochs_trained": len(history.history['accuracy'])
    }
    
    metrics_json_path = os.path.join(RESULTS_DIR, 'metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"--> Saved Evaluation Metrics JSON to {metrics_json_path}")
    print("\n[SUCCESS] CNN training and evaluation completed successfully!")

if __name__ == '__main__':
    train_and_evaluate()
