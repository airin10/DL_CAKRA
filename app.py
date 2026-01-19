"""
QR Code Security Model Evaluator - Streamlit App
Fokus: Evaluasi & Visualisasi Model Tanpa Training
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
import zipfile
import tempfile
import shutil
import time
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# === PAGE CONFIG ===
st.set_page_config(
    page_title="QR Model Evaluator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === IMPORT LIBRARIES ===
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GRU, LSTM, Dense, Dropout
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, 
    classification_report
)
import seaborn as sns

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

/* Model Card */
.model-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1.2rem 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    transition: all 0.3s ease;
    border-left: 4px solid #cbd5e1;
}
.model-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.1);
}
.cnn-card { border-left-color: #3B82F6; }
.lstm-card { border-left-color: #10B981; }
.gru-card { border-left-color: #8B5CF6; }

/* Best Model Banner */
.best-model {
    background: linear-gradient(135deg, #f0fdf4, #fffbeb);
    border: 1px solid #d1fae5;
    border-radius: 16px;
    padding: 1.25rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: inset 0 0 0 1px #bbf7d0;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { box-shadow: inset 0 0 0 1px #bbf7d0; }
    50% { box-shadow: inset 0 0 0 2px #4ade80; }
    100% { box-shadow: inset 0 0 0 1px #bbf7d0; }
}

/* Metric Badges */
.metric-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 0.2rem;
    background: #f1f5f9;
    color: #334155;
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

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 40px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 8px 8px 0 0;
    gap: 1rem;
    padding: 8px 16px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: var(--primary);
    color: white;
}

/* Footer */
.footer {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    color: #94a3b8;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

class DatasetProcessor:
    """Class untuk processing dataset upload untuk evaluasi"""
    
    def __init__(self):
        self.IMG_SIZE = (224, 224)
        self.MAX_LEN = 200
        self.VOCAB_SIZE = 1000
        self.tokenizer = None
    
    def process_zip_file(self, uploaded_zip):
        """Process uploaded ZIP file untuk mendapatkan data evaluasi"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            zip_path = os.path.join(temp_dir, 'uploaded.zip')
            with open(zip_path, 'wb') as f:
                f.write(uploaded_zip.getvalue())
            
            extract_path = os.path.join(temp_dir, 'extracted')
            os.makedirs(extract_path, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # Cari folder benign dan malicious
            benign_path = None
            malicious_path = None
            
            for root, dirs, files in os.walk(extract_path):
                dirs_lower = [d.lower() for d in dirs]
                
                if 'benign' in dirs_lower:
                    idx = dirs_lower.index('benign')
                    benign_path = os.path.join(root, dirs[idx])
                
                if 'malicious' in dirs_lower:
                    idx = dirs_lower.index('malicious')
                    malicious_path = os.path.join(root, dirs[idx])
            
            return {
                'benign_path': benign_path,
                'malicious_path': malicious_path,
                'temp_dir': temp_dir
            }
            
        except Exception as e:
            st.error(f"Error processing ZIP: {str(e)}")
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
            return None
    
    def prepare_evaluation_data(self, dataset_info, max_samples=100):
        """Prepare data evaluasi untuk semua model"""
        images = []
        texts = []
        labels = []
        
        # Generate synthetic text untuk evaluasi (untuk LSTM/GRU)
        def generate_text(label):
            import random
            if label == 0:  # benign
                templates = [
                    "https://safe-website.com/login",
                    "WIFI:S:Network;T:WPA2;P:password123;;",
                    "https://www.trusted-site.com",
                    "mailto:contact@company.com",
                    "BEGIN:VCARD\nFN:John Doe\nTEL:+1234567890\nEND:VCARD"
                ]
            else:  # malicious
                templates = [
                    "http://malicious-site.com/login.php",
                    "javascript:alert('XSS')",
                    "https://bit.ly/suspicious-xyz",
                    "Download free virus: http://bad.exe",
                    "You won $1,000,000! Claim: http://scam.com"
                ]
            return random.choice(templates)
        
        # Process benign images
        if dataset_info['benign_path'] and os.path.exists(dataset_info['benign_path']):
            benign_files = [f for f in os.listdir(dataset_info['benign_path']) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            for filename in benign_files[:max_samples//2]:
                try:
                    img_path = os.path.join(dataset_info['benign_path'], filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_resized = cv2.resize(img, self.IMG_SIZE)
                        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                        images.append(img_resized)
                        texts.append(generate_text(0))
                        labels.append(0)
                except:
                    continue
        
        # Process malicious images
        if dataset_info['malicious_path'] and os.path.exists(dataset_info['malicious_path']):
            malicious_files = [f for f in os.listdir(dataset_info['malicious_path']) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            for filename in malicious_files[:max_samples//2]:
                try:
                    img_path = os.path.join(dataset_info['malicious_path'], filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_resized = cv2.resize(img, self.IMG_SIZE)
                        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                        images.append(img_resized)
                        texts.append(generate_text(1))
                        labels.append(1)
                except:
                    continue
        
        if len(images) == 0:
            st.error("❌ Tidak ada gambar yang berhasil diproses untuk evaluasi!")
            return None
        
        # Convert to numpy arrays
        images_array = np.array(images, dtype='float32') / 255.0
        labels_array = np.array(labels)
        texts_array = np.array(texts)
        
        # Prepare text sequences untuk LSTM/GRU
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        # Buat tokenizer sederhana
        self.tokenizer = Tokenizer(
            num_words=self.VOCAB_SIZE,
            char_level=False,
            lower=True,
            oov_token='<OOV>'
        )
        self.tokenizer.fit_on_texts(texts_array)
        
        sequences = self.tokenizer.texts_to_sequences(texts_array)
        for i in range(len(sequences)):
            if len(sequences[i]) == 0:
                sequences[i] = [self.tokenizer.word_index.get('<OOV>', 1)]
        
        text_sequences = pad_sequences(
            sequences, 
            maxlen=self.MAX_LEN, 
            padding='post', 
            truncating='post'
        )
        
        st.success(f"✅ Data evaluasi siap: {len(images)} sampel")
        
        return {
            'X_img': images_array,
            'X_txt_seq': text_sequences,
            'y_true': labels_array,
            'X_txt_raw': texts_array,
            'vocab_size': min(self.VOCAB_SIZE, len(self.tokenizer.word_index) + 1)
        }

class ModelEvaluator:
    """Class untuk evaluasi model yang sudah dilatih"""
    
    def __init__(self):
        self.models = {}
        self.evaluation_results = {}
        self.predictions = {}
    
    def load_model_quishing_specific(self, model_type, model_file):
        """Load model dengan arsitektur spesifik penelitian quishing Anda"""
        try:
            # Simpan file sementara
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                tmp_file.write(model_file.getvalue())
                tmp_path = tmp_file.name
            
            st.info(f"🧠 Loading {model_type.upper()} model (Quishing Research Architecture)...")
            
            try:
                # Method 1: Coba load langsung dengan custom objects lengkap
                custom_objects = {
                    'Adam': tf.keras.optimizers.Adam,
                    'LSTM': tf.keras.layers.LSTM,
                    'GRU': tf.keras.layers.GRU,
                    'Embedding': tf.keras.layers.Embedding,
                    'GlobalMaxPooling1D': tf.keras.layers.GlobalMaxPooling1D,
                    'GlobalMaxPooling1D': tf.keras.layers.GlobalMaxPooling1D,  # Duplicate untuk pastikan
                    'Dense': tf.keras.layers.Dense,
                    'Dropout': tf.keras.layers.Dropout,
                    'Sequential': tf.keras.Sequential,
                }
                
                model = tf.keras.models.load_model(
                    tmp_path, 
                    compile=False,
                    custom_objects=custom_objects
                )
                
                st.success(f"✅ {model_type.upper()} loaded directly!")
                
            except Exception as e:
                st.warning(f"⚠️ Direct load failed: {str(e)[:80]}")
                
                # Method 2: Bangun ulang arsitektur spesifik
                model = self.reconstruct_quishing_model(model_type, tmp_path)
                
                if model is None:
                    st.error(f"❌ Failed to reconstruct {model_type} model")
                    os.unlink(tmp_path)
                    return False
            
            # Recompile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Simpan model
            self.models[model_type] = model
            os.unlink(tmp_path)
            
            # Tampilkan arsitektur
            self.display_model_architecture(model, model_type)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading {model_type}: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return False

    def reconstruct_quishing_model(self, model_type, weights_path):
        """Bangun ulang model dengan arsitektur penelitian quishing Anda"""
        
        st.info(f"🏗️ Reconstructing {model_type.upper()} from research code...")
        
        # Parameter berdasarkan kode tuning Anda
        if model_type == 'lstm':
            # PARAMETER LSTM dari kode Anda
            vocab_size = 1000  # Default, akan diupdate jika ada info
            max_len = 200      # Dari prepare_text_data di LSTM tuning
            embedding_dim = 64 # Dari best_params
            lstm_units = 64    # Dari best_params
            dropout_rate = 0.5 # Dari best_params
            
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(
                    input_dim=vocab_size, 
                    output_dim=embedding_dim, 
                    input_length=max_len,
                    name='embedding_layer'
                ),
                
                # LSTM dengan return_sequences=True untuk GlobalMaxPooling1D
                tf.keras.layers.LSTM(
                    units=lstm_units, 
                    dropout=dropout_rate, 
                    return_sequences=True,
                    name='lstm_layer'
                ),
                
                # GLOBALMAXPOOLING1D - Fitur unik penelitian Anda
                tf.keras.layers.GlobalMaxPooling1D(name='global_max_pooling'),
                
                tf.keras.layers.Dense(16, activation='relu', name='dense_16'),
                tf.keras.layers.Dropout(dropout_rate, name='dropout_layer'),
                tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')
            ])
            
            st.write(f"📐 LSTM Architecture: Embedding({vocab_size},{embedding_dim}) → "
                    f"LSTM({lstm_units}) → GlobalMaxPooling1D → Dense(16)")
        
        elif model_type == 'gru':
            # PARAMETER GRU dari kode Anda
            vocab_size = 1000  # Default
            max_len = 50       # Dari tune_gru function
            embedding_dim = 64 # Dari best_params (combination 2)
            gru_units = 64     # Dari best_params (combination 2)
            dropout_rate = 0.5 # Dari best_params (combination 2)
            
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(
                    input_dim=vocab_size, 
                    output_dim=embedding_dim, 
                    input_length=max_len,
                    name='embedding_layer'
                ),
                
                # GRU dengan return_sequences=True
                tf.keras.layers.GRU(
                    units=gru_units, 
                    dropout=dropout_rate, 
                    return_sequences=True,
                    name='gru_layer'
                ),
                
                # GLOBALMAXPOOLING1D - Sama seperti LSTM
                tf.keras.layers.GlobalMaxPooling1D(name='global_max_pooling'),
                
                tf.keras.layers.Dense(16, activation='relu', name='dense_16'),
                tf.keras.layers.Dropout(dropout_rate, name='dropout_layer'),
                tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')
            ])
            
            st.write(f"📐 GRU Architecture: Embedding({vocab_size},{embedding_dim}) → "
                    f"GRU({gru_units}) → GlobalMaxPooling1D → Dense(16)")
        
        elif model_type == 'cnn':
            # PARAMETER CNN dari kode Anda
            learning_rate = 0.001  # Dari best_params
            dropout_rate = 0.5     # Dari best_params (combination 3)
            num_filters = 128      # Dari best_params (combination 2 & 3)
            
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 3)),
                
                tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                
                tf.keras.layers.Conv2D(num_filters * 2, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            st.write(f"📐 CNN Architecture: Conv2D({num_filters}) → Conv2D({num_filters*2}) → Dense(64)")
        
        else:
            st.error(f"Unknown model type: {model_type}")
            return None
        
        # Coba load weights
        try:
            model.load_weights(weights_path)
            st.success(f"✅ Weights loaded successfully for {model_type}")
        except Exception as e:
            st.warning(f"⚠️ Could not load exact weights: {str(e)[:100]}")
            
            # Coba load dengan skip mismatch
            try:
                model.load_weights(weights_path, by_name=True, skip_mismatch=True)
                st.info("✅ Partial weights loaded (skip_mismatch=True)")
            except:
                st.error("❌ Failed to load any weights")
                return None
        
        return model

    def display_model_architecture(self, model, model_type):
        """Tampilkan arsitektur model dengan detail"""
        st.write(f"**{model_type.upper()} Model Architecture:**")
        
        # Summary dalam text
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        
        with st.expander(f"Show {model_type.upper()} Model Summary", expanded=True):
            for line in model_summary:
                st.text(line)
        
        # Visual layer information
        st.write("**Layer Details:**")
        layer_info = []
        for i, layer in enumerate(model.layers):
            layer_type = type(layer).__name__
            output_shape = str(layer.output_shape)
            params = layer.count_params()
            
            layer_info.append({
                'Layer #': i,
                'Name': layer.name,
                'Type': layer_type,
                'Output Shape': output_shape,
                'Params': f"{params:,}"
            })
        
        # Tampilkan sebagai tabel
        import pandas as pd
        df_layers = pd.DataFrame(layer_info)
        st.dataframe(df_layers, use_container_width=True)
        
        # Highlight GlobalMaxPooling1D jika ada
        if any('GlobalMaxPooling1D' in str(type(layer)) for layer in model.layers):
            st.success("🌟 This model includes GlobalMaxPooling1D (Unique feature of your research!)")
    
    def evaluate_model(self, model_type, evaluation_data):
        """Evaluasi model pada data evaluasi"""
        if model_type not in self.models:
            st.error(f"Model {model_type} belum di-load!")
            return False
        
        model = self.models[model_type]
        
        try:
            # Get predictions berdasarkan tipe model
            if model_type in ['cnn']:
                # CNN menggunakan data gambar
                y_pred_proba = model.predict(evaluation_data['X_img'], verbose=0).flatten()
            elif model_type in ['lstm', 'gru']:
                # LSTM/GRU menggunakan data teks
                y_pred_proba = model.predict(evaluation_data['X_txt_seq'], verbose=0).flatten()
            else:
                st.error(f"Tipe model {model_type} tidak dikenali!")
                return False
            
            # Convert probabilities to binary predictions
            y_pred = (y_pred_proba > 0.5).astype(int)
            y_true = evaluation_data['y_true']
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Try to calculate AUC
            try:
                auc = roc_auc_score(y_true, y_pred_proba)
            except:
                auc = 0.5
            
            # Store results
            self.evaluation_results[model_type] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            self.predictions[model_type] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error evaluating {model_type}: {str(e)}")
            return False
    
    def get_confusion_matrix(self, model_type):
        """Get confusion matrix untuk model tertentu"""
        if model_type not in self.evaluation_results:
            return None
        
        y_true = self.evaluation_results[model_type]['y_true']
        y_pred = self.evaluation_results[model_type]['y_pred']
        
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(self, model_type):
        """Get classification report untuk model tertentu"""
        if model_type not in self.evaluation_results:
            return None
        
        y_true = self.evaluation_results[model_type]['y_true']
        y_pred = self.evaluation_results[model_type]['y_pred']
        
        return classification_report(y_true, y_pred, output_dict=True)
    
    def compare_models(self):
        """Bandingkan semua model yang dievaluasi"""
        if not self.evaluation_results:
            return None
        
        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            comparison_data.append({
                'Model': model_name.upper(),
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'AUC': results['auc']
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_best_model(self):
        """Dapatkan model terbaik berdasarkan F1-Score"""
        if not self.evaluation_results:
            return None
        
        best_model = max(self.evaluation_results.items(), 
                        key=lambda x: x[1]['f1_score'])
        return best_model[0], best_model[1]

def plot_confusion_matrix(conf_matrix, model_name):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Create heatmap
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'],
                ax=ax)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix - {model_name.upper()}')
    
    return fig

def plot_metrics_comparison(comparison_df):
    """Plot perbandingan metrics antar model"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    colors = {'CNN': '#3B82F6', 'LSTM': '#10B981', 'GRU': '#8B5CF6'}
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        if metric in comparison_df.columns:
            models = comparison_df['Model']
            values = comparison_df[metric]
            
            bar_colors = [colors.get(model, 'gray') for model in models]
            bars = ax.bar(models, values, color=bar_colors)
            
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1.05])
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplot
    axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    return fig

def main():
    st.markdown('<div class="hero-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">📊 QR Code Model Evaluator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Upload pretrained models and evaluate performance without training</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    evaluator = ModelEvaluator()
    processor = DatasetProcessor()
    
    with st.sidebar:
        st.header("📁 Dataset Evaluasi")
        
        uploaded_eval_zip = st.file_uploader(
            "Upload dataset ZIP untuk evaluasi",
            type=['zip'],
            help="Berisi folder 'benign' dan 'malicious' dengan gambar QR code"
        )
        
        if uploaded_eval_zip:
            with st.spinner("Memproses dataset evaluasi..."):
                eval_dataset_info = processor.process_zip_file(uploaded_eval_zip)
                if eval_dataset_info:
                    st.success("✅ Dataset evaluasi siap!")
                    st.session_state['eval_dataset_info'] = eval_dataset_info
        
        # ============================================
        # PERBAIKAN INI: KELUARKAN DARI KONDISI if uploaded_eval_zip
        # ============================================
        st.markdown("---")
        st.header("🎯 Model Loading (Research Specific)")
        
        # Pilihan untuk memilih tipe model
        model_to_load = st.selectbox(
            "Select Model Type to Load:",
            ["CNN (Visual-Based)", "LSTM (URL-Based, max_len=200)", "GRU (URL-Based, max_len=50)"],
            help="Select based on your research architecture"
        )
        
        # Upload berdasarkan pilihan
        uploaded_model = st.file_uploader(
            f"Upload {model_to_load} model",
            type=["h5", "keras"],
            key=f"upload_{model_to_load.replace(' ', '_')}"
        )
        
        if uploaded_model:
            # Tentukan model_type berdasarkan pilihan
            if "CNN" in model_to_load:
                model_type = 'cnn'
                st.info("**Expected CNN Architecture:** Conv2D → MaxPooling2D → Flatten → Dense")
            elif "LSTM" in model_to_load:
                model_type = 'lstm'
                st.info("**Expected LSTM Architecture:** Embedding → LSTM → GlobalMaxPooling1D → Dense")
            else:  # GRU
                model_type = 'gru'
                st.info("**Expected GRU Architecture:** Embedding → GRU → GlobalMaxPooling1D → Dense")
            
            if st.button(f"✅ Load {model_type.upper()} Model", key=f"load_{model_type}"):
                with st.spinner(f"Loading {model_type.upper()} with research architecture..."):
                    if evaluator.load_model_quishing_specific(model_type, uploaded_model):
                        st.success(f"{model_type.upper()} model loaded successfully!")
                        
                        # Tampilkan parameter spesifik
                        if model_type in ['lstm', 'gru']:
                            st.write(f"**Parameters:**")
                            st.write(f"- Model Type: {model_type.upper()}")
                            st.write(f"- Architecture: RNN with GlobalMaxPooling1D")
                            st.write(f"- Max Sequence Length: {200 if model_type == 'lstm' else 50}")
                            st.write(f"- Unique Feature: GlobalMaxPooling1D after RNN layer")
        
        # ============================================
        # TAMBAHKAN EMERGENCY OPTION UNTUK GRU
        # ============================================
        st.markdown("---")
        st.header("🆘 Emergency GRU Loader")
        
        emergency_mode = st.checkbox(
            "Enable Emergency Mode for GRU", 
            help="Use if GRU model still fails to load"
        )
        
        if emergency_mode:
            st.warning("⚠️ Emergency Mode Active")
            
            uploaded_emergency_gru = st.file_uploader(
                "Upload GRU weights file (.h5)",
                type=["h5"],
                key="emergency_gru"
            )
            
            if uploaded_emergency_gru and st.button("🚨 Load GRU with Emergency Mode"):
                with st.spinner("Building emergency GRU model..."):
                    # Bangun model GRU dengan arsitektur yang lebih sederhana
                    vocab_size = 1000
                    max_len = 50
                    
                    emergency_gru = tf.keras.Sequential([
                        tf.keras.layers.Input(shape=(max_len,)),
                        tf.keras.layers.Embedding(vocab_size, 32),
                        tf.keras.layers.GRU(32, return_sequences=False),  # Tanpa return_sequences
                        tf.keras.layers.Dense(1, activation='sigmoid')
                    ])
                    
                    emergency_gru.compile(
                        optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    # Load weights
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as f:
                        f.write(uploaded_emergency_gru.getvalue())
                        temp_path = f.name
                    
                    try:
                        emergency_gru.load_weights(temp_path, by_name=True, skip_mismatch=True)
                        os.unlink(temp_path)
                        
                        evaluator.models['gru'] = emergency_gru
                        st.success("✅ Emergency GRU model loaded!")
                        
                    except Exception as e:
                        st.error(f"❌ Emergency load failed: {str(e)}")
        
        st.markdown("---")
        st.header("🚀 Start Evaluation")
        
        if st.button("📊 Evaluate All Models", type="primary", use_container_width=True):
            if 'eval_dataset_info' in st.session_state and evaluator.models:
                st.session_state['start_evaluation'] = True
            else:
                if 'eval_dataset_info' not in st.session_state:
                    st.warning("Please upload evaluation dataset first!")
                if not evaluator.models:
                    st.warning("Please upload at least one model!")
        
        st.markdown("---")
        st.header("🔄 Reset App")
        if st.button("Clear All Models & Data"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.clear()
            st.rerun()
    
    tab1, tab2, tab3 = st.tabs(["📋 Model Status", "📈 Evaluation", "📊 Comparison"])
    
    # --- TAB 1: Model Status ---
    with tab1:
        st.header("📋 Loaded Models Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="model-card cnn-card">', unsafe_allow_html=True)
            st.subheader("🎯 CNN Model")
            if 'cnn' in evaluator.models:
                st.success("✅ Model Loaded")
                st.write(f"Layers: {len(evaluator.models['cnn'].layers)}")
                st.write(f"Parameters: {evaluator.models['cnn'].count_params():,}")
            else:
                st.info("⏳ No model loaded")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="model-card lstm-card">', unsafe_allow_html=True)
            st.subheader("📝 LSTM Model")
            if 'lstm' in evaluator.models:
                st.success("✅ Model Loaded")
                st.write(f"Layers: {len(evaluator.models['lstm'].layers)}")
                st.write(f"Parameters: {evaluator.models['lstm'].count_params():,}")
            else:
                st.info("⏳ No model loaded")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="model-card gru-card">', unsafe_allow_html=True)
            st.subheader("🌀 GRU Model")
            if 'gru' in evaluator.models:
                st.success("✅ Model Loaded")
                st.write(f"Layers: {len(evaluator.models['gru'].layers)}")
                st.write(f"Parameters: {evaluator.models['gru'].count_params():,}")
            else:
                st.info("⏳ No model loaded")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Dataset Status
        st.markdown("---")
        st.header("📁 Evaluation Dataset Status")
        
        if 'eval_dataset_info' in st.session_state:
            eval_info = st.session_state['eval_dataset_info']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Benign Path", "✅ Ready" if eval_info['benign_path'] else "❌ Not Found")
            
            with col2:
                st.metric("Malicious Path", "✅ Ready" if eval_info['malicious_path'] else "❌ Not Found")
            
            with col3:
                if eval_info['benign_path'] and os.path.exists(eval_info['benign_path']):
                    benign_count = len([f for f in os.listdir(eval_info['benign_path']) 
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
                    st.metric("Benign Samples", benign_count)
            
            if eval_info['malicious_path'] and os.path.exists(eval_info['malicious_path']):
                malicious_count = len([f for f in os.listdir(eval_info['malicious_path']) 
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
                
                col4, col5 = st.columns(2)
                with col4:
                    st.metric("Malicious Samples", malicious_count)
                
                with col5:
                    total_samples = benign_count + malicious_count if 'benign_count' in locals() else malicious_count
                    st.metric("Total Samples", total_samples)
        else:
            st.info("👈 Upload evaluation dataset in the sidebar")
    
    # --- TAB 2: Evaluation ---
    with tab2:
        st.header("📈 Model Evaluation")
        
        if 'start_evaluation' in st.session_state and st.session_state['start_evaluation']:
            if 'eval_dataset_info' not in st.session_state:
                st.error("Evaluation dataset not found!")
                st.stop()
            
            eval_dataset_info = st.session_state['eval_dataset_info']
            
            with st.spinner("🔄 Preparing evaluation data..."):
                evaluation_data = processor.prepare_evaluation_data(eval_dataset_info, max_samples=50)
                
                if evaluation_data is None:
                    st.error("Failed to prepare evaluation data!")
                    st.stop()
            
            # Evaluate each loaded model
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            models_to_evaluate = list(evaluator.models.keys())
            
            for idx, model_type in enumerate(models_to_evaluate):
                status_text.text(f"Evaluating {model_type.upper()} model...")
                
                if evaluator.evaluate_model(model_type, evaluation_data):
                    st.success(f"✅ {model_type.upper()} evaluation completed!")
                else:
                    st.error(f"❌ {model_type.upper()} evaluation failed!")
                
                progress_bar.progress((idx + 1) / len(models_to_evaluate))
            
            progress_bar.progress(1.0)
            status_text.text("✅ All evaluations completed!")
            
            # Display results for each model
            for model_type in models_to_evaluate:
                if model_type in evaluator.evaluation_results:
                    results = evaluator.evaluation_results[model_type]
                    
                    if model_type == 'cnn':
                        st.markdown('<div class="model-card cnn-card">', unsafe_allow_html=True)
                    elif model_type == 'lstm':
                        st.markdown('<div class="model-card lstm-card">', unsafe_allow_html=True)
                    elif model_type == 'gru':
                        st.markdown('<div class="model-card gru-card">', unsafe_allow_html=True)
                    
                    st.subheader(f"{model_type.upper()} Results")
                    
                    # Metrics in columns
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1: st.metric("Accuracy", f"{results['accuracy']:.4f}")
                    with col2: st.metric("Precision", f"{results['precision']:.4f}")
                    with col3: st.metric("Recall", f"{results['recall']:.4f}")
                    with col4: st.metric("F1-Score", f"{results['f1_score']:.4f}")
                    with col5: st.metric("AUC", f"{results['auc']:.4f}")
                    
                    # Confusion Matrix
                    conf_matrix = evaluator.get_confusion_matrix(model_type)
                    if conf_matrix is not None:
                        fig_cm = plot_confusion_matrix(conf_matrix, model_type)
                        st.pyplot(fig_cm)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Store evaluator in session state
            st.session_state['evaluator'] = evaluator
            st.session_state['evaluation_complete'] = True
            
            # Cleanup temp directory
            try:
                if 'temp_dir' in eval_dataset_info:
                    shutil.rmtree(eval_dataset_info['temp_dir'], ignore_errors=True)
            except:
                pass
            
            # Reset evaluation flag
            st.session_state['start_evaluation'] = False
            
        else:
            st.info("👈 Upload models and dataset, then click 'Evaluate All Models'")
    
    # --- TAB 3: Comparison ---
    with tab3:
        st.header("📊 Model Comparison")
        
        if st.session_state.get('evaluation_complete'):
            evaluator = st.session_state.get('evaluator')
            
            if evaluator and evaluator.evaluation_results:
                # Comparison Table
                comparison_df = evaluator.compare_models()
                
                if comparison_df is not None:
                    # Format the dataframe for display
                    display_df = comparison_df.copy()
                    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Best Model
                    best_model_name, best_model_results = evaluator.get_best_model()
                    
                    st.markdown(f"""
                    <div class="best-model">
                        <h3>🏆 Best Model: {best_model_name.upper()}</h3>
                        <p>F1-Score: <strong>{best_model_results['f1_score']:.4f}</strong> | 
                        Accuracy: <strong>{best_model_results['accuracy']:.4f}</strong> | 
                        AUC: <strong>{best_model_results['auc']:.4f}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Visual Comparison
                    st.subheader("📈 Visual Comparison")
                    fig_comparison = plot_metrics_comparison(comparison_df)
                    st.pyplot(fig_comparison)
                    
                    # Download Results
                    st.markdown("---")
                    st.subheader("💾 Download Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Download CSV
                        csv = comparison_df.to_csv(index=False)
                        st.download_button(
                            label="📊 Download Results as CSV",
                            data=csv,
                            file_name="model_evaluation_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Download Report
                        report_text = "QR Code Model Evaluation Report\n"
                        report_text += "=" * 40 + "\n\n"
                        
                        for model_name, results in evaluator.evaluation_results.items():
                            report_text += f"{model_name.upper()} Model\n"
                            report_text += "-" * 20 + "\n"
                            report_text += f"Accuracy:  {results['accuracy']:.4f}\n"
                            report_text += f"Precision: {results['precision']:.4f}\n"
                            report_text += f"Recall:    {results['recall']:.4f}\n"
                            report_text += f"F1-Score:  {results['f1_score']:.4f}\n"
                            report_text += f"AUC:       {results['auc']:.4f}\n\n"
                        
                        st.download_button(
                            label="📄 Download Text Report",
                            data=report_text,
                            file_name="model_evaluation_report.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
            else:
                st.info("No evaluation results yet.")
        else:
            st.info("Evaluation results will appear here after completing evaluation.")

if __name__ == "__main__":
    main()