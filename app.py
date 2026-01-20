"""
QR Code Security Analyzer - MODERN UI VERSION dengan CNN, LSTM, GRU
[DIMODIFIKASI: CNN langsung upload model .h5 dengan MULTIPLE loading methods]
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
import h5py
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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
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
.qr-preview {
    background: white;
    border: 1px dashed #cbd5e1;
    border-radius: 12px;
    padding: 0.75rem;
    margin-top: 0.5rem;
    font-family: monospace;
    font-size: 0.85rem;
    max-height: 80px;
    overflow: auto;
    word-break: break-all;
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
.lstm-card { border-left-color: #10B981; }
.gru-card { border-left-color: #8B5CF6; }

/* Upload Section */
.upload-section {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 2px dashed #cbd5e1;
}

/* Metric badges */
.metric-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 0.2rem;
}
.metric-good { background: #dcfce7; color: #166534; }
.metric-medium { background: #fef9c3; color: #854d0e; }
.metric-poor { background: #fee2e2; color: #991b1b; }

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
        try:
            indices = np.arange(len(images_array))
            
            # Split untuk CNN
            train_idx_cnn, test_idx_cnn = train_test_split(
                indices,
                test_size=0.2,
                random_state=42,
                stratify=labels_array if len(indices) >= 10 and min(label_counts) >= 2 else None
            )
            
            # Split untuk text models
            train_idx_txt, test_idx_txt = train_test_split(
                indices,
                test_size=0.2,
                random_state=43
            )
            
        except Exception as e:
            # Fallback: manual split
            split_idx = int(len(images_array) * 0.8)
            train_idx_cnn = np.arange(split_idx)
            test_idx_cnn = np.arange(split_idx, len(images_array))
            train_idx_txt = train_idx_cnn.copy()
            test_idx_txt = test_idx_cnn.copy()
        
        # Prepare CNN data
        X_train_img = images_array[train_idx_cnn]
        X_test_img = images_array[test_idx_cnn]
        y_train_img = labels_array[train_idx_cnn]
        y_test_img = labels_array[test_idx_cnn]
        
        # Prepare text data
        X_train_txt = texts_array[train_idx_txt]
        X_test_txt = texts_array[test_idx_txt]
        y_train_txt = labels_array[train_idx_txt]
        y_test_txt = labels_array[test_idx_txt]
        
        # === PREPARE TEXT TOKENIZER ===
        try:
            self.tokenizer = Tokenizer(
                num_words=self.VOCAB_SIZE,
                char_level=False,
                lower=True,
                oov_token='<OOV>',
                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
            )
            
            self.tokenizer.fit_on_texts(X_train_txt)
            
            X_train_seq = self.texts_to_sequences(X_train_txt)
            X_test_seq = self.texts_to_sequences(X_test_txt)
            
            vocab_size = min(self.VOCAB_SIZE, len(self.tokenizer.word_index) + 1)
            
        except Exception as e:
            # Buat data dummy jika text processing gagal
            X_train_seq = np.zeros((len(X_train_txt), self.MAX_LEN))
            X_test_seq = np.zeros((len(X_test_txt), self.MAX_LEN))
            vocab_size = self.VOCAB_SIZE
        
        # === CALCULATE CLASS WEIGHTS ===
        try:
            classes = np.unique(y_train_txt)
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y_train_txt
            )
            class_weight_dict = dict(zip(classes, class_weights))
        except Exception as e:
            class_weight_dict = {0: 1.0, 1: 1.0}
        
        return {
            # CNN data
            'X_train_img': X_train_img,
            'X_test_img': X_test_img,
            'y_train_img': y_train_img,
            'y_test_img': y_test_img,
            
            # Text data
            'X_train_seq': X_train_seq,
            'X_test_seq': X_test_seq,
            'y_train_txt': y_train_txt,
            'y_test_txt': y_test_txt,
            'X_train_txt': X_train_txt,
            'X_test_txt': X_test_txt,
            
            # Info
            'total_samples': len(images),
            'valid_texts': valid_text_count,
            'class_weights': class_weight_dict,
            'vocab_size': vocab_size,
            'label_distribution': dict(zip(unique_labels, label_counts))
        }
    
    def texts_to_sequences(self, texts):
        """Convert texts to padded sequences"""
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        for i in range(len(sequences)):
            if len(sequences[i]) == 0:
                sequences[i] = [self.tokenizer.word_index.get('<OOV>', 1)]
        
        padded = pad_sequences(
            sequences, 
            maxlen=self.MAX_LEN, 
            padding='post', 
            truncating='post',
            value=self.tokenizer.word_index.get('<OOV>', 0)
        )
        
        return padded

class CompleteModelTrainer:
    """Complete trainer untuk semua model - MODIFIED: CNN hanya evaluasi dengan multiple loading methods"""
    
    def __init__(self):
        self.models = {}
        self.histories = {}
        self.results = {}
        self.cnn_model = None
    
    def load_cnn_model_with_custom_objects(self, model_file):
        """Load CNN model dengan custom objects untuk handle batch_shape error"""
        try:
            # Method 1: Coba load dengan custom_objects
            model = tf.keras.models.load_model(
                model_file,
                custom_objects={
                    'batch_shape': None,
                    'MobileNetV2': MobileNetV2,
                    'Adam': Adam
                }
            )
            return model
        except Exception as e:
            st.warning(f"Method 1 failed: {e}")
            return None
    
    def load_cnn_model_simple(self, model_file):
        """Load model dengan approach yang lebih sederhana"""
        try:
            # Method 2: Load weights saja, lalu bangun arsitektur
            with h5py.File(model_file, 'r') as f:
                # Cek jika file berisi model atau hanya weights
                if 'model_weights' in f or 'weights' in f:
                    # Ini adalah file model Keras
                    model = tf.keras.models.load_model(
                        model_file,
                        compile=False  # Don't compile saat load
                    )
                    
                    # Compile ulang dengan optimizer sederhana
                    model.compile(
                        optimizer=Adam(learning_rate=0.001),
                        loss='binary_crossentropy',
                        metrics=['accuracy', 'Precision', 'Recall']
                    )
                    return model
                else:
                    st.error("File tidak berisi model Keras yang valid")
                    return None
        except Exception as e:
            st.warning(f"Method 2 failed: {e}")
            return None
    
    def build_cnn_from_scratch_and_load_weights(self, model_file):
        """Bangun model dari scratch dan load weights"""
        try:
            # Method 3: Bangun model sama seperti di Colab
            base_model = MobileNetV2(
                weights='imagenet', 
                include_top=False, 
                input_shape=(224, 224, 3)
            )
            base_model.trainable = False
            
            model = Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(1, activation='sigmoid')
            ])
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'Precision', 'Recall', 'AUC']
            )
            
            # Load weights dari file
            model.load_weights(model_file)
            return model
            
        except Exception as e:
            st.warning(f"Method 3 failed: {e}")
            return None
    
    def load_cnn_model(self, model_file):
        """Try multiple methods to load CNN model"""
        st.info("🔄 Mencoba berbagai metode untuk load CNN model...")
        
        methods = [
            ("Metode 1: Standard load", lambda: tf.keras.models.load_model(model_file)),
            ("Metode 2: Load dengan custom objects", 
             lambda: self.load_cnn_model_with_custom_objects(model_file)),
            ("Metode 3: Load weights saja", 
             lambda: self.load_cnn_model_simple(model_file)),
            ("Metode 4: Bangun ulang & load weights", 
             lambda: self.build_cnn_from_scratch_and_load_weights(model_file)),
        ]
        
        for method_name, method_func in methods:
            try:
                st.write(f"🔍 Mencoba {method_name}...")
                model = method_func()
                if model:
                    st.success(f"✅ {method_name} berhasil!")
                    self.cnn_model = model
                    
                    # Tampilkan model summary
                    model_summary = []
                    model.summary(print_fn=lambda x: model_summary.append(x))
                    st.text_area("📋 Model Architecture", "\n".join(model_summary[:20]), height=200)
                    
                    return model
            except Exception as e:
                st.warning(f"❌ {method_name} gagal: {str(e)[:100]}")
                continue
        
        st.error("❌ Semua metode gagal load model. Pastikan file model valid.")
        return None
    
    def evaluate_cnn(self, training_data, model_file=None):
        """Evaluasi CNN model yang sudah dilatih"""
        st.markdown('<div class="training-card cnn-card">', unsafe_allow_html=True)
        st.subheader("🎯 Evaluating Pre-trained CNN Model")
        
        if model_file is None and self.cnn_model is None:
            st.error("❌ No CNN model provided for evaluation")
            st.markdown('</div>', unsafe_allow_html=True)
            return None
        
        # Load model jika belum diload
        if model_file and not self.cnn_model:
            model = self.load_cnn_model(model_file)
            if model is None:
                st.markdown('</div>', unsafe_allow_html=True)
                return None
        else:
            model = self.cnn_model
        
        with st.spinner("Evaluating CNN model..."):
            try:
                # Evaluate
                eval_results = model.evaluate(
                    training_data['X_test_img'], training_data['y_test_img'], verbose=0
                )
                
                # Handle different metric names
                if len(eval_results) >= 4:
                    # [loss, accuracy, precision, recall, ...]
                    metrics_dict = {
                        'loss': eval_results[0],
                        'accuracy': eval_results[1],
                        'precision': eval_results[2],
                        'recall': eval_results[3]
                    }
                    if len(eval_results) > 4:
                        metrics_dict['auc'] = eval_results[4]
                else:
                    # Fallback: hanya accuracy
                    metrics_dict = {
                        'loss': eval_results[0],
                        'accuracy': eval_results[1],
                        'precision': 0,
                        'recall': 0,
                        'auc': 0
                    }
                
                # Calculate F1-score
                f1 = 2 * (metrics_dict['precision'] * metrics_dict['recall']) / \
                     (metrics_dict['precision'] + metrics_dict['recall'] + 1e-7)
                
                self.models['cnn'] = model
                
                self.results['cnn'] = {
                    'accuracy': metrics_dict['accuracy'],
                    'precision': metrics_dict['precision'],
                    'recall': metrics_dict['recall'],
                    'auc': metrics_dict.get('auc', 0),
                    'loss': metrics_dict['loss'],
                    'f1_score': f1
                }
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{self.results['cnn']['accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{self.results['cnn']['precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{self.results['cnn']['recall']:.4f}")
                with col4:
                    st.metric("F1-Score", f"{self.results['cnn']['f1_score']:.4f}")
                
            except Exception as e:
                st.error(f"❌ Error evaluating CNN: {e}")
                self.results['cnn'] = {
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'auc': 0,
                    'loss': 0,
                    'f1_score': 0
                }
        
        st.markdown('</div>', unsafe_allow_html=True)
        return model
    
    def build_lstm_model(self, vocab_size, max_len=200):
        """Build LSTM model"""
        model = Sequential([
            layers.Embedding(
                input_dim=vocab_size,
                output_dim=64,
                input_length=max_len,
                mask_zero=True
            ),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2)),
            layers.Bidirectional(layers.LSTM(32, dropout=0.2)),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall', 'AUC']
        )
        
        return model
    
    def build_gru_model(self, vocab_size, max_len=200):
        """Build GRU model"""
        model = Sequential([
            layers.Embedding(
                input_dim=vocab_size,
                output_dim=64,
                input_length=max_len,
                mask_zero=True
            ),
            layers.Bidirectional(layers.GRU(64, return_sequences=True, dropout=0.2)),
            layers.Bidirectional(layers.GRU(32, dropout=0.2)),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall', 'AUC']
        )
        
        return model
    
    def train_lstm(self, training_data, epochs=15, batch_size=32):
        """Train LSTM model"""
        st.markdown('<div class="training-card lstm-card">', unsafe_allow_html=True)
        st.subheader("📝 Training LSTM")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        model = self.build_lstm_model(training_data['vocab_size'])
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)
        
        class LSTMCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {logs.get('loss', 0):.4f}, "
                    f"Acc: {logs.get('accuracy', 0):.4f}"
                )
        
        # Training
        with st.spinner("Training LSTM..."):
            history = model.fit(
                training_data['X_train_seq'], training_data['y_train_txt'],
                validation_data=(training_data['X_test_seq'], training_data['y_test_txt']),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=training_data.get('class_weights'),
                verbose=0,
                callbacks=[LSTMCallback(), early_stopping, reduce_lr]
            )
        
        progress_bar.progress(1.0)
        status_text.text("✅ LSTM Training Complete!")
        
        # Evaluate
        eval_results = model.evaluate(
            training_data['X_test_seq'], training_data['y_test_txt'], verbose=0
        )
        
        metrics_dict = dict(zip(model.metrics_names, eval_results))
        
        self.models['lstm'] = model
        self.histories['lstm'] = history.history
        
        # Calculate F1-score
        f1 = 2 * (metrics_dict['precision'] * metrics_dict['recall']) / \
             (metrics_dict['precision'] + metrics_dict['recall'] + 1e-7)
        
        self.results['lstm'] = {
            'accuracy': metrics_dict['accuracy'],
            'precision': metrics_dict['precision'],
            'recall': metrics_dict['recall'],
            'auc': metrics_dict['auc'],
            'loss': metrics_dict['loss'],
            'f1_score': f1
        }
        
        # Display results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{self.results['lstm']['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{self.results['lstm']['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{self.results['lstm']['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{self.results['lstm']['f1_score']:.4f}")
        
        # Plot training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history.history['accuracy'], label='Training')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('LSTM Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(history.history['loss'], label='Training')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('LSTM Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        return model
    
    def train_gru(self, training_data, epochs=15, batch_size=32):
        """Train GRU model"""
        st.markdown('<div class="training-card gru-card">', unsafe_allow_html=True)
        st.subheader("🌀 Training GRU")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        model = self.build_gru_model(training_data['vocab_size'])
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)
        
        class GRUCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {logs.get('loss', 0):.4f}, "
                    f"Acc: {logs.get('accuracy', 0):.4f}"
                )
        
        # Training
        with st.spinner("Training GRU..."):
            history = model.fit(
                training_data['X_train_seq'], training_data['y_train_txt'],
                validation_data=(training_data['X_test_seq'], training_data['y_test_txt']),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=training_data.get('class_weights'),
                verbose=0,
                callbacks=[GRUCallback(), early_stopping, reduce_lr]
            )
        
        progress_bar.progress(1.0)
        status_text.text("✅ GRU Training Complete!")
        
        # Evaluate
        eval_results = model.evaluate(
            training_data['X_test_seq'], training_data['y_test_txt'], verbose=0
        )
        
        metrics_dict = dict(zip(model.metrics_names, eval_results))
        
        self.models['gru'] = model
        self.histories['gru'] = history.history
        
        # Calculate F1-score
        f1 = 2 * (metrics_dict['precision'] * metrics_dict['recall']) / \
             (metrics_dict['precision'] + metrics_dict['recall'] + 1e-7)
        
        self.results['gru'] = {
            'accuracy': metrics_dict['accuracy'],
            'precision': metrics_dict['precision'],
            'recall': metrics_dict['recall'],
            'auc': metrics_dict['auc'],
            'loss': metrics_dict['loss'],
            'f1_score': f1
        }
        
        # Display results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{self.results['gru']['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{self.results['gru']['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{self.results['gru']['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{self.results['gru']['f1_score']:.4f}")
        
        # Plot training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history.history['accuracy'], label='Training')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('GRU Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(history.history['loss'], label='Training')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('GRU Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        return model
    
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
                'F1-Score': results['f1_score'],
                'AUC': results['auc'],
                'Loss': results['loss']
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_best_model(self):
        """Dapatkan model terbaik berdasarkan accuracy"""
        if not self.results:
            return None
        
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        return best_model[0], best_model[1]


def main():
    st.markdown('<div class="hero-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">🤖 QR Code Model Trainer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Upload pre-trained CNN model (.h5) + Train LSTM and GRU on your QR dataset</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    processor = DatasetProcessor()
    trainer = CompleteModelTrainer()

    with st.sidebar:
        st.header("📁 Dataset Upload")
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
        st.header("🎯 Model Selection")
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📤 Upload CNN Model")
        cnn_model_file = st.file_uploader(
            "Upload pre-trained CNN model (.h5)",
            type=['h5', 'keras'],
            key="cnn_uploader",
            help="Upload model yang sudah dipersiapkan dengan prepare_model_for_streamlit()"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("🤖 Select Models to Train")
        selected_models = st.multiselect(
            "Models to train/evaluate:",
            ["CNN (Upload Pre-trained)", "LSTM", "GRU"],
            default=["CNN (Upload Pre-trained)", "LSTM", "GRU"]
        )

        st.markdown("---")
        st.header("⚙️ Training Parameters")
        use_synthetic = st.checkbox("Use Synthetic Text Data", value=True,
                                    help="Generate synthetic QR text if decoding fails")
        epochs = st.slider("LSTM/GRU Epochs", 5, 30, 10)
        lstm_batch = st.slider("LSTM/GRU Batch Size", 16, 64, 32)

        if st.button("🚀 Train/Evaluate Models", type="primary", use_container_width=True):
            if 'dataset_info' in st.session_state:
                if "CNN (Upload Pre-trained)" in selected_models and not cnn_model_file:
                    st.warning("Please upload CNN model file first!")
                else:
                    st.session_state['selected_models'] = selected_models
                    st.session_state['cnn_model_file'] = cnn_model_file
                    st.session_state['training_params'] = {
                        'epochs': epochs,
                        'lstm_batch': lstm_batch,
                        'use_synthetic': use_synthetic
                    }
                    st.session_state['start_training'] = True
            else:
                st.warning("Please upload dataset first!")

        st.markdown("---")
        st.header("📝 Model Preparation Info")
        with st.expander("How to prepare models in Colab"):
            st.code('''
# DI COLAB - SEBELUM SAVE MODEL
def prepare_model_for_streamlit(model, model_name):
    """Prepare model agar kompatibel dengan Streamlit"""
    
    # 1. Clone model (remove optimizer state)
    model_copy = tf.keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())
    
    # 2. Recompile dengan optimizer sederhana
    model_copy.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # 3. Save as H5
    filename = f"{model_name}_streamlit_ready.h5"
    model_copy.save(filename, save_format='h5')
    
    return filename
            ''', language='python')

        st.markdown("---")
        st.header("🔄 Reset App")
        if st.button("Reset Cache & App State"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.clear()
            st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🤖 Training", "📈 Results", "💾 Download"])

    # --- TAB 1: Dataset ---
    with tab1:
        st.header("📊 Dataset Information")
        if 'dataset_info' in st.session_state:
            dataset_info = st.session_state['dataset_info']
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Images", dataset_info['total_images'])
            with col2: st.metric("Benign", dataset_info['benign_count'])
            with col3: st.metric("Malicious", dataset_info['malicious_count'])
            with col4:
                ratio = dataset_info['benign_count'] / dataset_info['total_images'] * 100 if dataset_info['total_images'] > 0 else 0
                st.metric("Benign Ratio", f"{ratio:.1f}%")

            if dataset_info['benign_samples']:
                st.subheader("✅ Benign Samples")
                cols = st.columns(min(5, len(dataset_info['benign_samples'])))
                for idx, img_path in enumerate(dataset_info['benign_samples']):
                    with cols[idx]:
                        try:
                            img = Image.open(img_path)
                            img.thumbnail((150, 150))
                            st.image(img, caption=f"Benign {idx+1}")
                        except: pass

            if dataset_info['malicious_samples']:
                st.subheader("❌ Malicious Samples")
                cols = st.columns(min(5, len(dataset_info['malicious_samples'])))
                for idx, img_path in enumerate(dataset_info['malicious_samples']):
                    with cols[idx]:
                        try:
                            img = Image.open(img_path)
                            img.thumbnail((150, 150))
                            st.image(img, caption=f"Malicious {idx+1}")
                        except: pass
        else:
            st.info("👈 Upload dataset ZIP file in the sidebar")

    # --- TAB 2: Training ---
    with tab2:
        st.header("🤖 Model Training & Evaluation")
        if 'start_training' in st.session_state and st.session_state['start_training']:
            if 'dataset_info' not in st.session_state:
                st.error("Dataset not found")
                return
            
            dataset_info = st.session_state['dataset_info']
            selected_models = st.session_state.get('selected_models', [])
            params = st.session_state.get('training_params', {})
            cnn_model_file = st.session_state.get('cnn_model_file')
            
            # Clear previous training state
            if 'training_complete' in st.session_state:
                st.session_state['training_complete'] = False
            
            with st.spinner("🔄 Preparing training data..."):
                try:
                    training_data = processor.prepare_training_data(
                        dataset_info,
                        use_synthetic=params['use_synthetic']
                    )
                    
                    if training_data is None:
                        st.error("❌ Failed to prepare training data")
                        st.stop()
                    
                    st.success(f"✅ Prepared {training_data['total_samples']} samples")
                    st.info(f"📝 Vocabulary size: {training_data['vocab_size']}")
                    st.info(f"⚖️ Label distribution: {training_data['label_distribution']}")
                    
                except Exception as e:
                    st.error(f"❌ Error in prepare_training_data: {e}")
                    st.stop()
            
            # Train/Evaluate selected models
            if "CNN (Upload Pre-trained)" in selected_models and cnn_model_file:
                # Save uploaded model to temp file
                temp_model_path = "temp_cnn_model.h5"
                with open(temp_model_path, "wb") as f:
                    f.write(cnn_model_file.getbuffer())
                
                try:
                    trainer.evaluate_cnn(training_data, temp_model_path)
                except Exception as e:
                    st.error(f"❌ CNN evaluation failed: {e}")
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_model_path):
                        os.remove(temp_model_path)
            
            if "LSTM" in selected_models:
                if training_data['vocab_size'] > 1:
                    try:
                        trainer.train_lstm(training_data, 
                                        epochs=params['epochs'], 
                                        batch_size=params['lstm_batch'])
                    except Exception as e:
                        st.error(f"❌ LSTM training failed: {e}")
                else:
                    st.warning("⚠️ No text data for LSTM training")
            
            if "GRU" in selected_models:
                if training_data['vocab_size'] > 1:
                    try:
                        trainer.train_gru(training_data, 
                                        epochs=params['epochs'], 
                                        batch_size=params['lstm_batch'])
                    except Exception as e:
                        st.error(f"❌ GRU training failed: {e}")
                else:
                    st.warning("⚠️ No text data for GRU training")
            
            # Mark training complete if at least one model processed
            if trainer.results:
                st.session_state['training_complete'] = True
                st.session_state['trainer'] = trainer
                st.balloons()
                
                # Cleanup temp directory
                if 'temp_dir' in dataset_info:
                    try:
                        shutil.rmtree(dataset_info['temp_dir'], ignore_errors=True)
                    except:
                        pass
            else:
                st.error("❌ No models were successfully processed")
            
            # Reset training flag
            st.session_state['start_training'] = False
            
        else:
            st.info("👈 Select models and click 'Train/Evaluate Models' in the sidebar")

    # --- TAB 3: Results ---
    with tab3:
        st.header("📈 Model Comparison")
        if st.session_state.get('training_complete'):
            trainer = st.session_state.get('trainer')
            if trainer and trainer.results:
                comparison_df = trainer.compare_models()
                if comparison_df is not None:
                    display_df = comparison_df.copy()
                    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
                    display_df['Loss'] = display_df['Loss'].apply(lambda x: f"{x:.6f}")
                    st.dataframe(display_df, use_container_width=True)

                    best_model_name, best_model_results = trainer.get_best_model()
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f0fdf4, #fffbeb);
                                border: 1px solid #d1fae5;
                                border-radius: 16px;
                                padding: 1.25rem;
                                text-align: center;
                                margin: 1.5rem 0;">
                        <h3>🏆 Best Model: {best_model_name.upper()}</h3>
                        <p>Accuracy: <strong>{best_model_results['accuracy']:.4f}</strong> |
                        F1-Score: <strong>{best_model_results['f1_score']:.4f}</strong> |
                        AUC: <strong>{best_model_results.get('auc', 0):.4f}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

                # Visual comparison
                st.subheader("📈 Visual Comparison")
                if trainer.results:
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    models = list(trainer.results.keys())
                    metrics_to_plot = [
                        ('Accuracy', 'accuracy'),
                        ('Precision', 'precision'),
                        ('Recall', 'recall'),
                        ('F1-Score', 'f1_score'),
                        ('AUC', 'auc'),
                        ('Loss', 'loss')
                    ]
                    colors = {'cnn': '#3B82F6', 'lstm': '#10B981', 'gru': '#8B5CF6'}
                    for idx, (title, metric_key) in enumerate(metrics_to_plot):
                        ax = axes[idx // 3, idx % 3]
                        values = [trainer.results[m].get(metric_key, 0) for m in models]
                        bar_colors = [colors.get(m, 'gray') for m in models]
                        bars = ax.bar(models, values, color=bar_colors)
                        ax.set_title(title)
                        ax.set_ylabel('Score' if metric_key != 'loss' else 'Loss')
                        ax.set_ylim([0, 1] if metric_key != 'loss' else [0, max(values) * 1.1])
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{height:.3f}', ha='center', va='bottom')
                    plt.tight_layout()
                    st.pyplot(fig)

                # Training history (hanya untuk LSTM dan GRU)
                st.subheader("📊 Training History")
                if trainer.histories:
                    fig, axes = plt.subplots(len(trainer.histories), 2, figsize=(12, 4 * len(trainer.histories)))
                    if len(trainer.histories) == 1: 
                        axes = [axes]
                    
                    for idx, (model_name, history) in enumerate(trainer.histories.items()):
                        if model_name in ['lstm', 'gru']:  # Hanya plot untuk LSTM dan GRU
                            ax1 = axes[idx][0]
                            ax1.plot(history['accuracy'], label='Training', linewidth=2)
                            if 'val_accuracy' in history:
                                ax1.plot(history['val_accuracy'], label='Validation', linewidth=2)
                            ax1.set_title(f'{model_name.upper()} - Accuracy')
                            ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.legend(); ax1.grid(True, alpha=0.3)

                            ax2 = axes[idx][1]
                            ax2.plot(history['loss'], label='Training', linewidth=2)
                            if 'val_loss' in history:
                                ax2.plot(history['val_loss'], label='Validation', linewidth=2)
                            ax2.set_title(f'{model_name.upper()} - Loss')
                            ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("No training results yet.")
        else:
            st.info("Training results will appear here after training completes.")

    # --- TAB 4: Download ---
    with tab4:
        st.header("💾 Download Models")
        if st.session_state.get('training_complete'):
            trainer = st.session_state.get('trainer')
            if trainer and trainer.models:
                os.makedirs('models', exist_ok=True)
                saved_files = {}
                
                # Save trained models (LSTM dan GRU saja)
                for model_name, model in trainer.models.items():
                    if model_name != 'cnn':  # Skip CNN karena sudah ada dari upload
                        filename = f"models/{model_name}_model.h5"
                        model.save(filename)
                        saved_files[model_name] = filename

                st.subheader("📥 Download Trained Models")
                cols = st.columns(3)
                model_display_names = {'lstm': "LSTM", 'gru': "GRU"}
                for idx, (key, name) in enumerate(model_display_names.items()):
                    if key in saved_files:
                        with cols[idx]:
                            with open(saved_files[key], 'rb') as f:
                                st.download_button(
                                    label=f"Download {name}",
                                    data=f.read(),
                                    file_name=f"{key}_model.h5",
                                    mime="application/octet-stream",
                                    use_container_width=True
                                )

                if trainer.results:
                    st.markdown("---")
                    st.subheader("📄 Download Results")
                    csv = trainer.compare_models().to_csv(index=False)
                    st.download_button(
                        label="📊 Download Results as CSV",
                        data=csv,
                        file_name="model_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.info("No models available for download.")
        else:
            st.info("Complete training to download models.")

# === RUN ===
if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    main()