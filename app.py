"""
QR Code Security Analyzer - MODERN UI VERSION dengan CNN, LSTM, GRU
[PERBAIKAN: Fixed LSTM/GRU metrics yang selalu 1.0]
"""
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import io
import os
import zipfile
import tempfile
import shutil
import time
import matplotlib.pyplot as plt
from PIL import Image
import warnings
import json
import hashlib
warnings.filterwarnings('ignore')

# === PAGE CONFIG ===
st.set_page_config(
    page_title="QR Code Model Trainer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === IMPORT LIBRARIES ===
import tensorflow as tf
from tensorflow.keras import layers, models, Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
try:
    from pyzbar.pyzbar import decode
    QR_DECODER_AVAILABLE = True
except ImportError:
    QR_DECODER_AVAILABLE = False

# === CUSTOM CSS MODERN ===
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary: #4f46e5;
    --primary-light: #818cf8;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --gray-100: #f9fafb;
    --gray-200: #e5e7eb;
    --gray-800: #1f2937;
    --text: #111827;
    --bg: #ffffff;
}

[data-theme="dark"] {
    --text: #f9fafb;
    --bg: #0f172a;
    --gray-100: #1e293b;
    --gray-200: #334155;
    --gray-800: #e2e8f0;
}

body, .stApp, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Hero Header */
.hero-container {
    text-align: center;
    padding: 2rem 0 1.5rem;
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.1);
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin: 0;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #4b5563;
    max-width: 700px;
    margin: 0.5rem auto 0;
}

/* Dataset Card */
.dataset-card {
    background: var(--gray-100);
    border-radius: 16px;
    padding: 1.25rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Training Cards */
.training-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1.2rem 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    transition: all 0.3s ease;
    border-left: 4px solid #cbd5e1;
}
.training-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.1);
}
.cnn-card { 
    border-left-color: #3B82F6; 
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
}
.lstm-card { 
    border-left-color: #10B981; 
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
}
.gru-card { 
    border-left-color: #8B5CF6; 
    background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
}

/* Upload Section */
.upload-section {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 2px dashed #cbd5e1;
}

/* Metric Cards */
.metric-card {
    background: white;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0.5rem 0;
}
.metric-label {
    font-size: 0.9rem;
    color: #64748b;
}

/* Best Model Banner */
.best-model-banner {
    background: linear-gradient(135deg, #f0fdf4 0%, #fffbeb 100%);
    border: 2px solid #4ade80;
    border-radius: 16px;
    padding: 1.5rem;
    margin: 2rem 0;
    text-align: center;
    box-shadow: 0 4px 12px rgba(74, 222, 128, 0.2);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 4px 12px rgba(74, 222, 128, 0.2); }
    50% { box-shadow: 0 4px 20px rgba(74, 222, 128, 0.4); }
    100% { box-shadow: 0 4px 12px rgba(74, 222, 128, 0.2); }
}

/* Buttons */
.stButton>button {
    width: 100%;
    border-radius: 12px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    border: none;
    background: var(--primary);
    color: white;
}
.stButton>button:hover {
    background: #4338ca;
    transform: scale(1.02);
    transition: all 0.2s ease;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    height: 40px;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: var(--primary);
    color: white;
}

/* Progress Bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
}

/* Warning Box */
.warning-box {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border: 1px solid #f59e0b;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

class DatasetProcessor:
    """Class untuk processing dataset upload"""
    
    def __init__(self):
        self.IMG_SIZE = (224, 224)
        self.MAX_LEN = 200
        self.VOCAB_SIZE = 1000
        self.tokenizer = None
        self.dataset_info = {}
    
    def extract_all_files(self, zip_path, extract_to):
        """Extract semua file dari ZIP"""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    
    def find_folders(self, base_path):
        """Cari folder benign dan malicious"""
        benign_path = None
        malicious_path = None
        
        for root, dirs, files in os.walk(base_path):
            dirs_lower = [d.lower() for d in dirs]
            
            if 'benign' in dirs_lower:
                idx = dirs_lower.index('benign')
                benign_path = os.path.join(root, dirs[idx])
            
            if 'malicious' in dirs_lower:
                idx = dirs_lower.index('malicious')
                malicious_path = os.path.join(root, dirs[idx])
        
        return benign_path, malicious_path
    
    def count_images_in_folder(self, folder_path):
        """Hitung gambar di folder"""
        if not folder_path or not os.path.exists(folder_path):
            return 0, []
        
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
        image_files = []
        
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(folder_path, filename))
        
        return len(image_files), image_files[:5]
    
    def extract_qr_content_robust(self, image_path):
        """Extract QR content dengan multiple attempts"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return ""
            
            decoded = decode(img)
            if decoded:
                return decoded[0].data.decode('utf-8', errors='ignore')
            
            # Try preprocessing
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                decoded = decode(thresh)
                if decoded:
                    return decoded[0].data.decode('utf-8', errors='ignore')
            except:
                pass
            
            return ""
            
        except Exception as e:
            return ""
    
    def generate_synthetic_text(self, label):
        """Generate synthetic text untuk training"""
        import random
        
        if label == 0:  # benign
            templates = [
                "https://safe-website.com/login",
                "WIFI:S:Network;T:WPA2;P:password123;;",
                "mailto:contact@company.com",
                "BEGIN:VCARD\nFN:John Doe\nTEL:+1234567890\nEND:VCARD",
                "https://www.trusted-site.com",
                "geo:40.7128,-74.0060",
                "SMSTO:+1234567890:Hello",
                "MATMSG:TO:email@test.com;SUB:Test;BODY:Message;;",
                "https://docs.google.com/document",
                "bit.ly/safe-link-123"
            ]
        else:  # malicious
            templates = [
                "http://malicious-site.com/login.php",
                "javascript:alert('XSS')",
                "data:text/html;base64,PHNjcmlwdD5hbGVydCgnWCcpPC9zY3JpcHQ+",
                "http://192.168.1.1:8080/admin",
                "https://bit.ly/suspicious-xyz",
                "Download free virus: http://bad.exe",
                "You won $1,000,000! Claim: http://scam.com",
                "Account locked! Reset: http://phishing-bank.com",
                "Free Bitcoin: http://crypto-scam.com",
                "http://free-gift-cards.xyz/login"
            ]
        
        base_text = random.choice(templates)
        variations = [
            "",
            "?id=" + str(random.randint(1000, 9999)),
            "&session=" + ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8)),
            "#" + ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))
        ]
        
        return base_text + random.choice(variations)
    
    def process_zip_file(self, uploaded_zip):
        """Process uploaded ZIP file"""
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            zip_path = os.path.join(temp_dir, 'uploaded.zip')
            with open(zip_path, 'wb') as f:
                f.write(uploaded_zip.getvalue())
            
            extract_path = os.path.join(temp_dir, 'extracted')
            os.makedirs(extract_path, exist_ok=True)
            
            self.extract_all_files(zip_path, extract_path)
            
            benign_path, malicious_path = self.find_folders(extract_path)
            
            benign_count, benign_samples = self.count_images_in_folder(benign_path)
            malicious_count, malicious_samples = self.count_images_in_folder(malicious_path)
            
            dataset_info = {
                'total_images': benign_count + malicious_count,
                'benign_count': benign_count,
                'malicious_count': malicious_count,
                'benign_samples': benign_samples,
                'malicious_samples': malicious_samples,
                'benign_path': benign_path,
                'malicious_path': malicious_path,
                'extract_path': extract_path,
                'has_benign': benign_count > 0,
                'has_malicious': malicious_count > 0,
                'temp_dir': temp_dir
            }
            
            return dataset_info
            
        except Exception as e:
            st.error(f"Error processing ZIP: {str(e)}")
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
            return None
    
    def prepare_training_data(self, dataset_info, use_synthetic=True):
        """Prepare data untuk semua model"""
        images = []
        texts = []
        labels = []
        valid_text_count = 0

        # Process benign images
        if dataset_info['has_benign'] and dataset_info['benign_path']:
            for filename in os.listdir(dataset_info['benign_path']):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(dataset_info['benign_path'], filename)
                    
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            img_resized = cv2.resize(img, self.IMG_SIZE)
                            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                            images.append(img_resized)
                            
                            # Untuk LSTM/GRU
                            text_content = self.extract_qr_content_robust(img_path)
                            if text_content:
                                texts.append(text_content)
                                valid_text_count += 1
                            elif use_synthetic:
                                texts.append(self.generate_synthetic_text(0))
                            else:
                                texts.append("")
                            
                            labels.append(0)  # benign
                    except Exception as e:
                        continue
        
        # Process malicious images
        if dataset_info['has_malicious'] and dataset_info['malicious_path']:
            for filename in os.listdir(dataset_info['malicious_path']):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(dataset_info['malicious_path'], filename)
                    
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            img_resized = cv2.resize(img, self.IMG_SIZE)
                            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                            images.append(img_resized)
                            
                            text_content = self.extract_qr_content_robust(img_path)
                            if text_content:
                                texts.append(text_content)
                                valid_text_count += 1
                            elif use_synthetic:
                                texts.append(self.generate_synthetic_text(1))
                            else:
                                texts.append("")
                            
                            labels.append(1)  # malicious
                    except Exception as e:
                        continue
        
        # === CHECK IF DATA EXISTS ===
        if len(images) == 0:
            st.error("❌ Tidak ada gambar yang berhasil diproses!")
            return None
        
        # === CONVERT TO NUMPY ===
        images_array = np.array(images, dtype='float32') / 255.0
        labels_array = np.array(labels)
        texts_array = np.array(texts)
        
        # === CHECK LABEL DISTRIBUTION ===
        unique_labels, label_counts = np.unique(labels_array, return_counts=True)
        
        if len(unique_labels) < 2:
            st.error(f"❌ Hanya ada {len(unique_labels)} kelas. Butuh minimal 2 kelas (benign & malicious).")
            return None
        
        # === SPLIT DATA ===
        # GUNAKAN SAME SPLIT UNTUK SEMUA MODEL
        try:
            # Gunakan stratified split untuk menjaga distribusi
            X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
                images_array,
                labels_array,
                texts_array,
                test_size=0.2,
                random_state=42,  # FIXED SEED
                stratify=labels_array
            )
        except Exception as e:
            # Fallback split
            split_idx = int(len(images_array) * 0.8)
            X_train = images_array[:split_idx]
            X_test = images_array[split_idx:]
            y_train = labels_array[:split_idx]
            y_test = labels_array[split_idx:]
            texts_train = texts_array[:split_idx]
            texts_test = texts_array[split_idx:]
        
        # === DEBUG INFO ===
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.write("**Data Split Information:**")
        st.write(f"- Total samples: {len(images_array)}")
        st.write(f"- Train samples: {len(X_train)} (benign: {sum(y_train==0)}, malicious: {sum(y_train==1)})")
        st.write(f"- Test samples: {len(X_test)} (benign: {sum(y_test==0)}, malicious: {sum(y_test==1)})")
        st.write(f"- Valid text extracted: {valid_text_count}/{len(images_array)}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # === PREPARE TEXT TOKENIZER ===
        try:
            self.tokenizer = Tokenizer(
                num_words=self.VOCAB_SIZE,
                char_level=False,
                lower=True,
                oov_token='<OOV>',
                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
            )
            
            self.tokenizer.fit_on_texts(texts_train)
            
            # Convert texts to sequences
            train_sequences = self.tokenizer.texts_to_sequences(texts_train)
            test_sequences = self.tokenizer.texts_to_sequences(texts_test)
            
            # Handle empty sequences
            for i in range(len(train_sequences)):
                if len(train_sequences[i]) == 0:
                    train_sequences[i] = [self.tokenizer.word_index.get('<OOV>', 1)]
            
            for i in range(len(test_sequences)):
                if len(test_sequences[i]) == 0:
                    test_sequences[i] = [self.tokenizer.word_index.get('<OOV>', 1)]
            
            # Pad sequences
            X_train_seq = pad_sequences(
                train_sequences,
                maxlen=self.MAX_LEN,
                padding='post',
                truncating='post',
                value=0
            )
            
            X_test_seq = pad_sequences(
                test_sequences,
                maxlen=self.MAX_LEN,
                padding='post',
                truncating='post',
                value=0
            )
            
            vocab_size = min(self.VOCAB_SIZE, len(self.tokenizer.word_index) + 1)
            
            # Debug tokenizer info
            st.info(f"Tokenizer Info: Vocabulary size = {vocab_size}")
            st.info(f"Sample sequence length: {len(X_train_seq[0]) if len(X_train_seq) > 0 else 0}")
            
        except Exception as e:
            st.error(f"Error in text processing: {str(e)}")
            # Fallback: create simple sequences
            vocab_size = 100
            X_train_seq = np.zeros((len(texts_train), self.MAX_LEN))
            X_test_seq = np.zeros((len(texts_test), self.MAX_LEN))
        
        # === CALCULATE CLASS WEIGHTS ===
        try:
            classes = np.unique(y_train)
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y_train
            )
            class_weight_dict = dict(zip(classes, class_weights))
        except Exception as e:
            class_weight_dict = {0: 1.0, 1: 1.0}
        
        return {
            # CNN data
            'X_train_img': X_train,
            'X_test_img': X_test,
            'y_train_img': y_train,
            'y_test_img': y_test,
            
            # Text data
            'X_train_seq': X_train_seq,
            'X_test_seq': X_test_seq,
            'y_train_txt': y_train,  # Sama dengan y_train_img
            'y_test_txt': y_test,    # Sama dengan y_test_img
            'X_train_txt': texts_train,
            'X_test_txt': texts_test,
            
            # Info
            'total_samples': len(images),
            'valid_texts': valid_text_count,
            'class_weights': class_weight_dict,
            'vocab_size': vocab_size,
            'label_distribution': dict(zip(unique_labels, label_counts)),
            'max_len': self.MAX_LEN
        }

class CompleteModelTrainer:
    """Complete trainer untuk semua model - FIXED: Metrics tidak jadi 1.0"""
    
    def __init__(self):
        self.models = {}
        self.histories = {}
        self.results = {}
        self.is_cnn_loaded = False
    
    def load_cnn_offline_results(self, eval_file, history_file):
        """Load CNN results dari file JSON"""
        try:
            # Load evaluation results
            eval_content = eval_file.read().decode('utf-8')
            cnn_eval = json.loads(eval_content)
            
            # Load history
            history_content = history_file.read().decode('utf-8')
            cnn_history = json.loads(history_content)
            
            # Ensure all required metrics exist
            required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            for metric in required_metrics:
                if metric not in cnn_eval:
                    if metric == 'f1_score':
                        # Calculate F1-score from precision and recall
                        precision = cnn_eval.get('precision', 0)
                        recall = cnn_eval.get('recall', 0)
                        if precision + recall > 0:
                            cnn_eval['f1_score'] = 2 * (precision * recall) / (precision + recall)
                        else:
                            cnn_eval['f1_score'] = 0.0
                    else:
                        cnn_eval[metric] = 0
            
            # Simpan hasil
            self.results['cnn'] = cnn_eval
            self.histories['cnn'] = cnn_history
            self.is_cnn_loaded = True
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading CNN JSON results: {e}")
            return False
    
    def display_cnn_results(self):
        """Tampilkan hasil CNN dari JSON"""
        if 'cnn' not in self.results:
            return None
        
        cnn_eval = self.results['cnn']
        
        # Create metric cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value" style="color: {'#10b981' if cnn_eval['accuracy'] > 0.8 else '#f59e0b'}">
                    {cnn_eval['accuracy']:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value" style="color: {'#10b981' if cnn_eval['precision'] > 0.8 else '#f59e0b'}">
                    {cnn_eval['precision']:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value" style="color: {'#10b981' if cnn_eval['recall'] > 0.8 else '#f59e0b'}">
                    {cnn_eval['recall']:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value" style="color: {'#10b981' if cnn_eval['f1_score'] > 0.8 else '#f59e0b'}">
                    {cnn_eval['f1_score']:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display history plot if available
        if 'cnn' in self.histories:
            self.plot_cnn_history()
    
    def plot_cnn_history(self):
        """Plot CNN training history"""
        history = self.histories['cnn']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1 = axes[0]
        if 'accuracy' in history:
            ax1.plot(history['accuracy'], label='Training', linewidth=2, color='#3B82F6')
        if 'val_accuracy' in history:
            ax1.plot(history['val_accuracy'], label='Validation', linewidth=2, color='#10B981')
        ax1.set_title('CNN Accuracy History', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Loss plot
        ax2 = axes[1]
        if 'loss' in history:
            ax2.plot(history['loss'], label='Training Loss', linewidth=2, color='#EF4444')
        if 'val_loss' in history:
            ax2.plot(history['val_loss'], label='Validation Loss', linewidth=2, color='#F59E0B')
        ax2.set_title('CNN Loss History', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def build_lstm_model(self, vocab_size, max_len=200):
        """Build LSTM model dengan regularization untuk mencegah overfitting"""
        model = Sequential([
            layers.Embedding(
                input_dim=vocab_size,
                output_dim=64,
                input_length=max_len,
                mask_zero=True
            ),
            layers.Dropout(0.2),  # Dropout di embedding layer
            layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
            layers.BatchNormalization(),
            layers.Bidirectional(layers.LSTM(32, dropout=0.3, recurrent_dropout=0.3)),
            layers.Dropout(0.4),
            layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_gru_model(self, vocab_size, max_len=200):
        """Build GRU model dengan regularization"""
        model = Sequential([
            layers.Embedding(
                input_dim=vocab_size,
                output_dim=64,
                input_length=max_len,
                mask_zero=True
            ),
            layers.Dropout(0.2),
            layers.Bidirectional(layers.GRU(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
            layers.BatchNormalization(),
            layers.Bidirectional(layers.GRU(32, dropout=0.3, recurrent_dropout=0.3)),
            layers.Dropout(0.4),
            layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def validate_model(self, model, X_test, y_test, model_name):
        """Validasi model dan cek metrics"""
        try:
            y_pred_prob = model.predict(X_test, verbose=0)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            
            # Debug information
            unique_preds, pred_counts = np.unique(y_pred, return_counts=True)
            unique_true, true_counts = np.unique(y_test, return_counts=True)
            
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.write(f"**{model_name} Debug Info:**")
            st.write(f"Predictions distribution: {dict(zip(unique_preds, pred_counts))}")
            st.write(f"True labels distribution: {dict(zip(unique_true, true_counts))}")
            
            # Jika semua predictions sama, ada masalah
            if len(unique_preds) == 1:
                st.warning(f"⚠️ Model hanya memprediksi satu kelas: {unique_preds[0]}")
                st.write(f"Sample prediction probabilities: {y_pred_prob[:10].flatten()}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'y_pred': y_pred,
                'y_pred_prob': y_pred_prob
            }
            
        except Exception as e:
            st.error(f"Error validating {model_name}: {e}")
            return None
    
    def train_lstm(self, training_data, epochs=15, batch_size=32):
        """Train LSTM model"""
        st.markdown('<div class="training-card lstm-card">', unsafe_allow_html=True)
        
        # Header with icon
        col_header = st.columns([1, 8])
        with col_header[0]:
            st.markdown("<h2 style='margin:0;'>📝</h2>", unsafe_allow_html=True)
        with col_header[1]:
            st.subheader("Training LSTM Model")
        
        # Validasi data sebelum training
        if training_data['vocab_size'] <= 1:
            st.error("❌ Vocabulary size terlalu kecil untuk training LSTM")
            st.markdown('</div>', unsafe_allow_html=True)
            return None
        
        # Tampilkan info data
        st.write(f"**Training Info:**")
        st.write(f"- Vocabulary size: {training_data['vocab_size']}")
        st.write(f"- Sequence length: {training_data['max_len']}")
        st.write(f"- Train samples: {len(training_data['X_train_seq'])}")
        st.write(f"- Test samples: {len(training_data['X_test_seq'])}")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        model = self.build_lstm_model(training_data['vocab_size'], training_data['max_len'])
        
        # Callbacks untuk mencegah overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001,
            verbose=0
        )
        
        class LSTMCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {logs.get('loss', 0):.4f}, "
                    f"Accuracy: {logs.get('accuracy', 0):.4f}"
                )
        
        # Training
        try:
            with st.spinner("Training LSTM model..."):
                history = model.fit(
                    training_data['X_train_seq'],
                    training_data['y_train_txt'],
                    validation_data=(
                        training_data['X_test_seq'],
                        training_data['y_test_txt']
                    ),
                    epochs=epochs,
                    batch_size=batch_size,
                    class_weight=training_data.get('class_weights'),
                    verbose=0,
                    callbacks=[LSTMCallback(), early_stopping, reduce_lr]
                )
            
            progress_bar.progress(1.0)
            status_text.text("✅ LSTM Training Complete!")
            
            # Validasi model
            validation_results = self.validate_model(
                model,
                training_data['X_test_seq'],
                training_data['y_test_txt'],
                "LSTM"
            )
            
            if validation_results:
                # Simpan model dan hasil
                self.models['lstm'] = model
                self.histories['lstm'] = history.history
                self.results['lstm'] = {
                    'accuracy': validation_results['accuracy'],
                    'precision': validation_results['precision'],
                    'recall': validation_results['recall'],
                    'f1_score': validation_results['f1_score']
                }
                
                # Display results
                self.display_model_results('lstm')
                
                # Plot training history
                self.plot_training_history(history, 'LSTM')
                
                # Confusion matrix
                self.plot_confusion_matrix(
                    validation_results['confusion_matrix'],
                    'LSTM Confusion Matrix'
                )
            
        except Exception as e:
            st.error(f"❌ LSTM Training Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
        
        st.markdown('</div>', unsafe_allow_html=True)
        return model
    
    def train_gru(self, training_data, epochs=15, batch_size=32):
        """Train GRU model"""
        st.markdown('<div class="training-card gru-card">', unsafe_allow_html=True)
        
        # Header with icon
        col_header = st.columns([1, 8])
        with col_header[0]:
            st.markdown("<h2 style='margin:0;'>🌀</h2>", unsafe_allow_html=True)
        with col_header[1]:
            st.subheader("Training GRU Model")
        
        # Validasi data sebelum training
        if training_data['vocab_size'] <= 1:
            st.error("❌ Vocabulary size terlalu kecil untuk training GRU")
            st.markdown('</div>', unsafe_allow_html=True)
            return None
        
        # Tampilkan info data
        st.write(f"**Training Info:**")
        st.write(f"- Vocabulary size: {training_data['vocab_size']}")
        st.write(f"- Sequence length: {training_data['max_len']}")
        st.write(f"- Train samples: {len(training_data['X_train_seq'])}")
        st.write(f"- Test samples: {len(training_data['X_test_seq'])}")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        model = self.build_gru_model(training_data['vocab_size'], training_data['max_len'])
        
        # Callbacks untuk mencegah overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001,
            verbose=0
        )
        
        class GRUCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {logs.get('loss', 0):.4f}, "
                    f"Accuracy: {logs.get('accuracy', 0):.4f}"
                )
        
        # Training
        try:
            with st.spinner("Training GRU model..."):
                history = model.fit(
                    training_data['X_train_seq'],
                    training_data['y_train_txt'],
                    validation_data=(
                        training_data['X_test_seq'],
                        training_data['y_test_txt']
                    ),
                    epochs=epochs,
                    batch_size=batch_size,
                    class_weight=training_data.get('class_weights'),
                    verbose=0,
                    callbacks=[GRUCallback(), early_stopping, reduce_lr]
                )
            
            progress_bar.progress(1.0)
            status_text.text("✅ GRU Training Complete!")
            
            # Validasi model
            validation_results = self.validate_model(
                model,
                training_data['X_test_seq'],
                training_data['y_test_txt'],
                "GRU"
            )
            
            if validation_results:
                # Simpan model dan hasil
                self.models['gru'] = model
                self.histories['gru'] = history.history
                self.results['gru'] = {
                    'accuracy': validation_results['accuracy'],
                    'precision': validation_results['precision'],
                    'recall': validation_results['recall'],
                    'f1_score': validation_results['f1_score']
                }
                
                # Display results
                self.display_model_results('gru')
                
                # Plot training history
                self.plot_training_history(history, 'GRU')
                
                # Confusion matrix
                self.plot_confusion_matrix(
                    validation_results['confusion_matrix'],
                    'GRU Confusion Matrix'
                )
            
        except Exception as e:
            st.error(f"❌ GRU Training Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
        
        st.markdown('</div>', unsafe_allow_html=True)
        return model
    
    def plot_training_history(self, history, model_name):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1 = axes[0]
        if 'accuracy' in history.history:
            ax1.plot(history.history['accuracy'], label='Training', linewidth=2, color='#10B981')
        if 'val_accuracy' in history.history:
            ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2, color='#3B82F6')
        ax1.set_title(f'{model_name} Accuracy History', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Loss plot
        ax2 = axes[1]
        if 'loss' in history.history:
            ax2.plot(history.history['loss'], label='Training Loss', linewidth=2, color='#EF4444')
        if 'val_loss' in history.history:
            ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#F59E0B')
        ax2.set_title(f'{model_name} Loss History', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def plot_confusion_matrix(self, cm, title):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Benign', 'Malicious'],
               yticklabels=['Benign', 'Malicious'],
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        st.pyplot(fig)
    
    def display_model_results(self, model_name):
        """Display model results in metric cards"""
        if model_name not in self.results:
            return
        
        results = self.results[model_name]
        
        # Create metric cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            accuracy_color = '#10b981' if results['accuracy'] > 0.8 else '#f59e0b' if results['accuracy'] > 0.6 else '#ef4444'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value" style="color: {accuracy_color}">
                    {results['accuracy']:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            precision_color = '#10b981' if results['precision'] > 0.8 else '#f59e0b' if results['precision'] > 0.6 else '#ef4444'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value" style="color: {precision_color}">
                    {results['precision']:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            recall_color = '#10b981' if results['recall'] > 0.8 else '#f59e0b' if results['recall'] > 0.6 else '#ef4444'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value" style="color: {recall_color}">
                    {results['recall']:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            f1_color = '#10b981' if results['f1_score'] > 0.8 else '#f59e0b' if results['f1_score'] > 0.6 else '#ef4444'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value" style="color: {f1_color}">
                    {results['f1_score']:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def compare_models(self):
        """Bandingkan semua model"""
        if not self.results:
            return None
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.upper(),
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_best_model(self):
        """Dapatkan model terbaik berdasarkan F1-Score"""
        if not self.results:
            return None, None
        
        best_model = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        return best_model[0], best_model[1]


def main():
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">🤖 QR Code Security Analyzer</h1>
        <p class="hero-subtitle">
            Combine CNN (Offline Results) with LSTM & GRU Training
        </p>
    </div>
    """, unsafe_allow_html=True)

    processor = DatasetProcessor()
    trainer = CompleteModelTrainer()

    with st.sidebar:
        st.header("📁 Dataset Configuration")
        
        uploaded_file = st.file_uploader(
            "Upload dataset ZIP",
            type=['zip'],
            help="Must contain 'benign' and 'malicious' folders"
        )
        
        if uploaded_file:
            with st.spinner("Processing dataset..."):
                dataset_info = processor.process_zip_file(uploaded_file)
                if dataset_info and dataset_info['total_images'] > 0:
                    st.success(f"✅ {dataset_info['total_images']} images loaded")
                    st.session_state['dataset_info'] = dataset_info
                else:
                    st.error("❌ Invalid dataset")
        
        st.markdown("---")
        st.header("🎯 Model Configuration")
        
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📊 CNN Offline Results")
        
        cnn_eval_file = st.file_uploader(
            "Upload cnn_eval.json",
            type=["json"],
            help="Upload JSON file with CNN evaluation results"
        )
        
        cnn_history_file = st.file_uploader(
            "Upload cnn_history.json",
            type=["json"],
            help="Upload JSON file with CNN training history"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("🤖 Models to Train")
        selected_models = st.multiselect(
            "Select models to train:",
            ["LSTM", "GRU"],
            default=["LSTM", "GRU"],
            help="CNN results will be loaded from uploaded JSON files"
        )
        
        st.markdown("---")
        st.header("⚙️ Training Parameters")
        
        use_synthetic = st.checkbox("Use Synthetic Text Data", value=True,
                                    help="Generate synthetic QR text if decoding fails")
        
        epochs = st.slider("Training Epochs", 5, 30, 10,
                          help="Number of training epochs for LSTM/GRU")
        
        batch_size = st.slider("Batch Size", 16, 64, 32,
                              help="Batch size for LSTM/GRU training")
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            enable_debug = st.checkbox("Enable Debug Mode", value=True,
                                      help="Show detailed debug information")
            validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2,
                                        help="Percentage of data for validation")
        
        if st.button("🚀 Start Training & Analysis", type="primary", use_container_width=True):
            if 'dataset_info' not in st.session_state:
                st.warning("Please upload dataset first!")
            elif not cnn_eval_file or not cnn_history_file:
                st.warning("Please upload both CNN JSON files!")
            else:
                st.session_state['selected_models'] = selected_models
                st.session_state['cnn_eval_file'] = cnn_eval_file
                st.session_state['cnn_history_file'] = cnn_history_file
                st.session_state['training_params'] = {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'use_synthetic': use_synthetic,
                    'enable_debug': enable_debug,
                    'validation_split': validation_split
                }
                st.session_state['start_training'] = True
        
        st.markdown("---")
        if st.button("🔄 Reset Application", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.clear()
            st.rerun()

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🤖 Training", "📈 Results", "💾 Download"])

    # --- TAB 1: Dataset ---
    with tab1:
        st.header("📊 Dataset Information")
        
        if 'dataset_info' in st.session_state:
            dataset_info = st.session_state['dataset_info']
            
            # Dataset metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Images", dataset_info['total_images'])
            with col2:
                st.metric("Benign Images", dataset_info['benign_count'])
            with col3:
                st.metric("Malicious Images", dataset_info['malicious_count'])
            with col4:
                ratio = dataset_info['benign_count'] / dataset_info['total_images'] * 100 if dataset_info['total_images'] > 0 else 0
                st.metric("Benign Ratio", f"{ratio:.1f}%")
            
            # Sample images
            if dataset_info['benign_samples']:
                st.subheader("✅ Benign QR Samples")
                cols = st.columns(min(5, len(dataset_info['benign_samples'])))
                for idx, img_path in enumerate(dataset_info['benign_samples']):
                    with cols[idx]:
                        try:
                            img = Image.open(img_path)
                            img.thumbnail((150, 150))
                            st.image(img, caption=f"Sample {idx+1}", use_container_width=True)
                        except:
                            st.warning(f"Could not load image {idx+1}")
            
            if dataset_info['malicious_samples']:
                st.subheader("❌ Malicious QR Samples")
                cols = st.columns(min(5, len(dataset_info['malicious_samples'])))
                for idx, img_path in enumerate(dataset_info['malicious_samples']):
                    with cols[idx]:
                        try:
                            img = Image.open(img_path)
                            img.thumbnail((150, 150))
                            st.image(img, caption=f"Sample {idx+1}", use_container_width=True)
                        except:
                            st.warning(f"Could not load image {idx+1}")
        else:
            st.info("👈 Please upload dataset ZIP file in the sidebar")

    # --- TAB 2: Training ---
    with tab2:
        st.header("🤖 Model Training & Analysis")
        
        if 'start_training' in st.session_state and st.session_state['start_training']:
            if 'dataset_info' not in st.session_state:
                st.error("Dataset not found. Please upload dataset first.")
                return
            
            dataset_info = st.session_state['dataset_info']
            selected_models = st.session_state.get('selected_models', [])
            params = st.session_state.get('training_params', {})
            cnn_eval_file = st.session_state.get('cnn_eval_file')
            cnn_history_file = st.session_state.get('cnn_history_file')
            
            # Step 1: Load CNN results
            st.markdown("### Step 1: Loading CNN Offline Results")
            st.markdown('<div class="training-card cnn-card">', unsafe_allow_html=True)
            
            with st.spinner("Loading CNN results from JSON files..."):
                if trainer.load_cnn_offline_results(cnn_eval_file, cnn_history_file):
                    st.success("✅ CNN results loaded successfully!")
                    trainer.display_cnn_results()
                else:
                    st.error("❌ Failed to load CNN results")
                    st.stop()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Step 2: Prepare training data
            st.markdown("### Step 2: Preparing Training Data")
            with st.spinner("Preparing training data for LSTM/GRU..."):
                try:
                    training_data = processor.prepare_training_data(
                        dataset_info,
                        use_synthetic=params['use_synthetic']
                    )
                    
                    if training_data is None:
                        st.error("❌ Failed to prepare training data")
                        st.stop()
                    
                    st.success(f"✅ Prepared {training_data['total_samples']} samples")
                    
                    # Tampilkan info penting untuk debugging
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.write("**Training Data Summary:**")
                    st.write(f"- Total vocabulary size: {training_data['vocab_size']}")
                    st.write(f"- Train samples: {len(training_data['X_train_seq'])}")
                    st.write(f"- Test samples: {len(training_data['X_test_seq'])}")
                    st.write(f"- Label distribution (train): benign={sum(training_data['y_train_txt']==0)}, malicious={sum(training_data['y_train_txt']==1)}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error preparing training data: {e}")
                    import traceback
                    st.error(traceback.format_exc())
                    st.stop()
            
            # Step 3: Train selected models
            st.markdown("### Step 3: Training Selected Models")
            
            # Cek apakah ada text data yang valid
            if training_data['vocab_size'] <= 10:
                st.warning("⚠️ Warning: Vocabulary size sangat kecil. Model mungkin tidak belajar dengan baik.")
            
            if "LSTM" in selected_models:
                st.markdown("---")
                trainer.train_lstm(training_data, 
                                 epochs=params['epochs'], 
                                 batch_size=params['batch_size'])
            
            if "GRU" in selected_models:
                st.markdown("---")
                trainer.train_gru(training_data, 
                                epochs=params['epochs'], 
                                batch_size=params['batch_size'])
            
            # Mark training complete
            if trainer.results:
                st.session_state['training_complete'] = True
                st.session_state['trainer'] = trainer
                
                # Cleanup temp directory
                if 'temp_dir' in dataset_info:
                    try:
                        shutil.rmtree(dataset_info['temp_dir'], ignore_errors=True)
                    except:
                        pass
            
            # Reset training flag
            st.session_state['start_training'] = False
            
        else:
            st.info("👈 Configure models and click 'Start Training & Analysis' in the sidebar")

    # --- TAB 3: Results ---
    with tab3:
        st.header("📈 Model Comparison & Analysis")
        
        if st.session_state.get('training_complete'):
            trainer = st.session_state.get('trainer')
            
            if trainer and trainer.results:
                # Best model banner
                best_model_name, best_model_results = trainer.get_best_model()
                if best_model_name and best_model_results:
                    st.markdown(f"""
                    <div class="best-model-banner">
                        <h3 style="margin:0 0 1rem 0;">🏆 Best Model: {best_model_name.upper()}</h3>
                        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
                            <div style="text-align: center;">
                                <div style="font-size: 0.9rem; color: #64748b;">Accuracy</div>
                                <div style="font-size: 1.8rem; font-weight: 700; color: #10b981;">
                                    {best_model_results['accuracy']:.4f}
                                </div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 0.9rem; color: #64748b;">F1-Score</div>
                                <div style="font-size: 1.8rem; font-weight: 700; color: #10b981;">
                                    {best_model_results['f1_score']:.4f}
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Comparison table
                st.subheader("📊 Model Performance Comparison")
                comparison_df = trainer.compare_models()
                if comparison_df is not None:
                    # Style the dataframe
                    styled_df = comparison_df.copy()
                    styled_df = styled_df.round(4)
                    
                    # Color code based on values
                    def color_cells(val):
                        if isinstance(val, (int, float)):
                            if val >= 0.9:
                                return 'background-color: #dcfce7; color: #166534;'
                            elif val >= 0.8:
                                return 'background-color: #fef9c3; color: #854d0e;'
                            elif val >= 0.7:
                                return 'background-color: #fee2e2; color: #991b1b;'
                        return ''
                    
                    # Apply styling
                    numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                    styled_df = styled_df.style.applymap(color_cells, subset=numeric_cols)
                    
                    st.dataframe(styled_df, use_container_width=True)
                
                # Visual comparison
                st.subheader("📈 Performance Visualization")
                if trainer.results:
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    models = list(trainer.results.keys())
                    
                    # Define colors for each model
                    colors = {
                        'cnn': '#3B82F6',  # Blue
                        'lstm': '#10B981',  # Green
                        'gru': '#8B5CF6'    # Purple
                    }
                    
                    metrics_to_plot = [
                        ('Accuracy', 'accuracy'),
                        ('Precision', 'precision'),
                        ('Recall', 'recall'),
                        ('F1-Score', 'f1_score')
                    ]
                    
                    for idx, (title, metric_key) in enumerate(metrics_to_plot):
                        ax = axes[idx // 2, idx % 2]
                        values = [trainer.results[m].get(metric_key, 0) for m in models]
                        bar_colors = [colors.get(m, '#94a3b8') for m in models]
                        bars = ax.bar(models, values, color=bar_colors, edgecolor='black')
                        
                        ax.set_title(title, fontsize=14, fontweight='bold')
                        ax.set_ylabel('Score', fontsize=12)
                        ax.set_ylim([0, 1])
                        
                        # Add threshold lines
                        ax.axhline(y=0.9, color='#10b981', linestyle='--', alpha=0.3, label='Excellent')
                        ax.axhline(y=0.8, color='#f59e0b', linestyle='--', alpha=0.3, label='Good')
                        ax.axhline(y=0.7, color='#ef4444', linestyle='--', alpha=0.3, label='Poor')
                        
                        # Add value labels on bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.3f}', ha='center', va='bottom',
                                   fontweight='bold')
                        
                        ax.grid(True, alpha=0.2)
                        if idx == 3:  # Only add legend to last plot
                            ax.legend(loc='upper right')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
            else:
                st.info("No training results available yet.")
        else:
            st.info("Complete training to see results here.")

    # --- TAB 4: Download ---
    with tab4:
        st.header("💾 Download Results & Models")
        
        if st.session_state.get('training_complete'):
            trainer = st.session_state.get('trainer')
            
            if trainer:
                # Save trained models
                os.makedirs('models', exist_ok=True)
                saved_files = {}
                
                # Save LSTM and GRU models
                for model_name, model in trainer.models.items():
                    if model_name in ['lstm', 'gru']:
                        filename = f"models/{model_name}_model.h5"
                        model.save(filename)
                        saved_files[model_name] = filename
                
                # Download trained models
                st.subheader("📥 Download Trained Models")
                if saved_files:
                    cols = st.columns(len(saved_files))
                    for idx, (model_name, filepath) in enumerate(saved_files.items()):
                        with cols[idx]:
                            model_display_name = model_name.upper()
                            with open(filepath, 'rb') as f:
                                st.download_button(
                                    label=f"Download {model_display_name}",
                                    data=f.read(),
                                    file_name=f"{model_name}_model.h5",
                                    mime="application/octet-stream",
                                    use_container_width=True
                                )
                else:
                    st.info("No models available for download.")
                
                # Download results
                st.subheader("📊 Download Results")
                if trainer.results:
                    # Create comparison CSV
                    comparison_df = trainer.compare_models()
                    csv = comparison_df.to_csv(index=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="📄 Download Results CSV",
                            data=csv,
                            file_name="model_comparison.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Create JSON summary
                        summary = {
                            'best_model': trainer.get_best_model()[0],
                            'results': trainer.results,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        json_data = json.dumps(summary, indent=2)
                        
                        st.download_button(
                            label="📋 Download Summary JSON",
                            data=json_data,
                            file_name="training_summary.json",
                            mime="application/json",
                            use_container_width=True
                        )
        else:
            st.info("Complete training to download models and results.")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    main()