"""
Script untuk training models
"""

import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from model_manager import ModelManager
import tensorflow as tf

# Disable GPU jika ada masalah
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_sample_data():
    """Create sample data untuk training"""
    print("Creating sample data...")
    
    # Buat folder
    os.makedirs('sample_data/benign', exist_ok=True)
    os.makedirs('sample_data/malicious', exist_ok=True)
    
    # Sample QR content
    benign_urls = [
        "https://google.com",
        "https://github.com",
        "https://wikipedia.org",
        "https://youtube.com",
        "https://facebook.com",
        "WIFI:S:MyNetwork;T:WPA;P:MyPassword;;",
        "BEGIN:VCARD\nVERSION:3.0\nFN:John Doe\nTEL:1234567890\nEND:VCARD",
        "Hello, this is a safe QR code",
        "Contact: email@example.com",
        "Location: Latitude, Longitude"
    ]
    
    malicious_urls = [
        "http://free-gift-card.com/login",
        "javascript:alert('malicious')",
        "data:text/html;base64,PHNjcmlwdD5hbGVydCgnWCcpPC9zY3JpcHQ+",
        "http://192.168.1.1:8080/login.php",
        "https://bit.ly/suspicious-link-123",
        "http://phishing-bank.com/verify",
        "Download this: http://malware.exe",
        "WINNER! Claim your prize: http://fake-lottery.com",
        "Your account will be locked: http://fake-security.com",
        "Click here for free bitcoin: http://scam-site.com"
    ]
    
    # Create dummy image data
    X_img = []
    X_txt = []
    y = []
    
    # Benign samples
    for i, text in enumerate(benign_urls[:5]):
        # Create simple QR-like image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        X_img.append(img)
        X_txt.append(text)
        y.append(0)  # benign
    
    # Malicious samples
    for i, text in enumerate(malicious_urls[:5]):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        X_img.append(img)
        X_txt.append(text)
        y.append(1)  # malicious
    
    return np.array(X_img), np.array(X_txt), np.array(y)

def main():
    """Main training function"""
    print("=" * 60)
    print("Training CNN, LSTM, and GRU Models")
    print("=" * 60)
    
    # Create sample data
    X_img, X_txt, y = create_sample_data()
    
    print(f"Dataset size: {len(X_img)} samples")
    print(f"Benign: {np.sum(y == 0)}, Malicious: {np.sum(y == 1)}")
    
    # Split data
    X_train_img, X_val_img, X_train_txt, X_val_txt, y_train, y_val = train_test_split(
        X_img, X_txt, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train_img)}")
    print(f"Validation samples: {len(X_val_img)}")
    
    # Initialize model manager
    manager = ModelManager()
    
    # Train models
    print("\n" + "=" * 60)
    print("Starting model training...")
    print("=" * 60)
    
    history = manager.train_models(
        X_train_img, X_train_txt, y_train,
        X_val_img, X_val_txt, y_val,
        epochs=5  # Untuk demo, gunakan epoch sedikit
    )
    
    print("\n✅ Training completed!")
    print("Models saved to 'models/' folder")
    
    # Display training results
    print("\n" + "=" * 60)
    print("Training Results Summary")
    print("=" * 60)
    
    for model_name, hist in history.items():
        print(f"\n📊 {model_name.upper()}:")
        if 'accuracy' in hist:
            final_acc = hist['accuracy'][-1]
            val_acc = hist['val_accuracy'][-1] if 'val_accuracy' in hist else None
            print(f"   Final Accuracy: {final_acc:.4f}")
            if val_acc:
                print(f"   Validation Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    # Set memory growth untuk TensorFlow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    main()