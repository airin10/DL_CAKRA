"""
QR Code Security Analyzer - DATASET PREVIEW + MODEL RESULTS
[Upload dataset untuk preview + Upload model results untuk analisis, NO TRAINING]
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
import cv2
import zipfile
import tempfile
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Evaluating CNN LSTM and GRU for Malicious QR Code Identification",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0.5rem 0;
}
.metric-label {
    font-size: 0.9rem;
    color: #64748b;
    font-weight: 500;
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

/* Info Box */
.info-box {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    border: 1px solid #3b82f6;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}

/* Warning Box */
.warning-box {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border: 1px solid #f59e0b;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}

/* Success Box */
.success-box {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    border: 1px solid #22c55e;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}

/* Image Container */
.image-container {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
    margin-bottom: 1rem;
}
.image-container:hover {
    transform: translateY(-5px);
}
</style>
""", unsafe_allow_html=True)

class DatasetPreview:
    """Class untuk preview dataset saja (tidak untuk training)"""
    
    def __init__(self):
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
            
            # Break early jika sudah ditemukan
            if benign_path and malicious_path:
                break
        
        return benign_path, malicious_path
    
    def count_images_in_folder(self, folder_path):
        """Hitung gambar di folder"""
        if not folder_path or not os.path.exists(folder_path):
            return 0, []
        
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp']
        image_files = []
        
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(folder_path, filename))
        
        # Ambil maksimal 10 sample untuk preview
        return len(image_files), image_files[:10]
    
    def get_image_info(self, image_path):
        """Get informasi tentang gambar"""
        try:
            img = Image.open(image_path)
            return {
                'size': img.size,
                'mode': img.mode,
                'format': img.format
            }
        except:
            return None
    
    def process_zip_file(self, uploaded_zip):
        """Process uploaded ZIP file untuk preview saja"""
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Simpan file ZIP
            zip_path = os.path.join(temp_dir, 'uploaded.zip')
            with open(zip_path, 'wb') as f:
                f.write(uploaded_zip.getvalue())
            
            # Extract ZIP
            extract_path = os.path.join(temp_dir, 'extracted')
            os.makedirs(extract_path, exist_ok=True)
            
            self.extract_all_files(zip_path, extract_path)
            
            # Cari folder benign dan malicious
            benign_path, malicious_path = self.find_folders(extract_path)
            
            # Hitung gambar di setiap folder
            benign_count, benign_samples = self.count_images_in_folder(benign_path)
            malicious_count, malicious_samples = self.count_images_in_folder(malicious_path)
            
            # Collect sample image info
            benign_sample_info = []
            for img_path in benign_samples[:5]:  # Ambil 5 sample untuk info
                info = self.get_image_info(img_path)
                if info:
                    benign_sample_info.append(info)
            
            malicious_sample_info = []
            for img_path in malicious_samples[:5]:
                info = self.get_image_info(img_path)
                if info:
                    malicious_sample_info.append(info)
            
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
                'temp_dir': temp_dir,
                'benign_sample_info': benign_sample_info,
                'malicious_sample_info': malicious_sample_info
            }
            
            return dataset_info
            
        except Exception as e:
            st.error(f"❌ Error processing ZIP: {str(e)}")
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
            return None
    
    def display_dataset_info(self, dataset_info):
        """Display dataset information untuk preview"""
        
        if not dataset_info:
            return
        
        st.write("**ℹ️ Dataset Preview Only** - Dataset tidak digunakan untuk training, hanya untuk preview.")
        
        # Dataset metrics card
        st.markdown('<div class="dataset-card">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Images", dataset_info['total_images'])
        with col2:
            st.metric("Benign Images", dataset_info['benign_count'])
        with col3:
            st.metric("Malicious Images", dataset_info['malicious_count'])
        with col4:
            if dataset_info['total_images'] > 0:
                ratio = (dataset_info['benign_count'] / dataset_info['total_images']) * 100
                st.metric("Benign Ratio", f"{ratio:.1f}%")
            else:
                st.metric("Benign Ratio", "0%")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Distribution chart
        if dataset_info['benign_count'] > 0 or dataset_info['malicious_count'] > 0:
            st.subheader("📊 Dataset Distribution")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            labels = ['Benign', 'Malicious']
            counts = [dataset_info['benign_count'], dataset_info['malicious_count']]
            colors = ['#10b981', '#ef4444']
            
            bars = ax.bar(labels, counts, color=colors, edgecolor='black')
            ax.set_ylabel('Number of Images', fontsize=12)
            ax.set_title('Dataset Class Distribution', fontsize=14, fontweight='bold')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}', ha='center', va='bottom',
                       fontweight='bold')
            
            # Add pie chart
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            if dataset_info['total_images'] > 0:
                ax2.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%',
                       startangle=90, wedgeprops={'edgecolor': 'black'})
                ax2.set_title('Dataset Proportion', fontsize=14, fontweight='bold')
            
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig)
            with col2:
                if dataset_info['total_images'] > 0:
                    st.pyplot(fig2)
            
            plt.close('all')
        
        # Dataset statistics
        st.markdown("### 📈 Dataset Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📋 File Information:**")
            st.write(f"- Total files: **{dataset_info['total_images']}**")
            st.write(f"- Benign files: **{dataset_info['benign_count']}**")
            st.write(f"- Malicious files: **{dataset_info['malicious_count']}**")
            
            if dataset_info['total_images'] > 0:
                st.write(f"- Balance ratio: **{dataset_info['benign_count']/dataset_info['total_images']:.2%}**")
            
            # Image format info jika ada sample
            if dataset_info['benign_sample_info'] or dataset_info['malicious_sample_info']:
                st.write("\n**🖼️ Sample Image Formats:**")
                
                all_formats = []
                if dataset_info['benign_sample_info']:
                    for info in dataset_info['benign_sample_info']:
                        if info and 'format' in info:
                            all_formats.append(info['format'])
                
                if dataset_info['malicious_sample_info']:
                    for info in dataset_info['malicious_sample_info']:
                        if info and 'format' in info:
                            all_formats.append(info['format'])
                
                if all_formats:
                    unique_formats = set(all_formats)
                    st.write(f"- Formats found: {', '.join(unique_formats)}")
        
        with col2:
            st.write("**⚖️ Quality Indicators:**")
            
            if dataset_info['benign_count'] == 0:
                st.error("⚠️ **Tidak ada gambar benign**")
            elif dataset_info['malicious_count'] == 0:
                st.error("⚠️ **Tidak ada gambar malicious**")
            elif abs(dataset_info['benign_count'] - dataset_info['malicious_count']) / dataset_info['total_images'] > 0.7:
                st.warning("⚠️ Dataset sangat tidak seimbang")
            else:
                st.success("✅ Dataset cukup seimbang")
            
            if dataset_info['total_images'] < 50:
                st.warning("⚠️ Dataset sangat kecil (<50 samples)")
            elif dataset_info['total_images'] < 200:
                st.info("ℹ️ Dataset ukuran sedang (50-200 samples)")
            elif dataset_info['total_images'] < 1000:
                st.success("✅ Dataset cukup besar (200-1000 samples)")
            else:
                st.success("✅ Dataset sangat besar (>1000 samples)")
        
        # Sample images - Benign
        if dataset_info['benign_samples']:
            st.markdown("---")
            st.subheader("✅ Benign QR Samples")
            
            # Display 5 samples in a row
            cols = st.columns(5)
            for idx, img_path in enumerate(dataset_info['benign_samples'][:5]):
                with cols[idx]:
                    try:
                        img = Image.open(img_path)
                        img.thumbnail((150, 150))
                        
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(img, caption=f"Benign {idx+1}", width=150)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # File info
                        file_size = os.path.getsize(img_path) / 1024  # KB
                        st.caption(f"{os.path.basename(img_path)[:15]}...\n{file_size:.1f} KB")
                        
                    except Exception as e:
                        st.warning(f"Could not load image {idx+1}")
            
            # Show more samples in expander
            if len(dataset_info['benign_samples']) > 5:
                with st.expander(f"Show more benign samples"):
                    # Calculate grid layout
                    num_cols = 5
                    num_rows = (len(dataset_info['benign_samples']) + num_cols - 1) // num_cols
                    
                    for row in range(num_rows):
                        cols = st.columns(num_cols)
                        for col in range(num_cols):
                            idx = row * num_cols + col
                            if idx < len(dataset_info['benign_samples']):
                                with cols[col]:
                                    try:
                                        img = Image.open(dataset_info['benign_samples'][idx])
                                        img.thumbnail((120, 120))
                                        st.image(img, caption=f"B{idx+1}", width=120)
                                    except:
                                        st.write(f"Image {idx+1}")
        
        # Sample images - Malicious
        if dataset_info['malicious_samples']:
            st.markdown("---")
            st.subheader("❌ Malicious QR Samples")
            
            # Display 5 samples in a row
            cols = st.columns(5)
            for idx, img_path in enumerate(dataset_info['malicious_samples'][:5]):
                with cols[idx]:
                    try:
                        img = Image.open(img_path)
                        img.thumbnail((150, 150))
                        
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(img, caption=f"Malicious {idx+1}", width=150)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # File info
                        file_size = os.path.getsize(img_path) / 1024  # KB
                        st.caption(f"{os.path.basename(img_path)[:15]}...\n{file_size:.1f} KB")
                        
                    except Exception as e:
                        st.warning(f"Could not load image {idx+1}")
            
            # Show more samples in expander
            if len(dataset_info['malicious_samples']) > 5:
                with st.expander(f"Show more malicious samples"):
                    # Calculate grid layout
                    num_cols = 5
                    num_rows = (len(dataset_info['malicious_samples']) + num_cols - 1) // num_cols
                    
                    for row in range(num_rows):
                        cols = st.columns(num_cols)
                        for col in range(num_cols):
                            idx = row * num_cols + col
                            if idx < len(dataset_info['malicious_samples']):
                                with cols[col]:
                                    try:
                                        img = Image.open(dataset_info['malicious_samples'][idx])
                                        img.thumbnail((120, 120))
                                        st.image(img, caption=f"M{idx+1}", width=120)
                                    except:
                                        st.write(f"Image {idx+1}")
        
        # Dataset summary
        st.markdown("---")
        st.markdown("### 📝 Dataset Summary")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.write("**✅ Strengths:**")
            if dataset_info['total_images'] >= 200:
                st.write("- Dataset cukup besar untuk training")
            if dataset_info['benign_count'] > 0 and dataset_info['malicious_count'] > 0:
                st.write("- Memiliki kedua kelas (benign & malicious)")
            if dataset_info['total_images'] > 0 and abs(dataset_info['benign_count'] - dataset_info['malicious_count']) / dataset_info['total_images'] <= 0.3:
                st.write("- Distribusi kelas cukup seimbang")
        
        with summary_col2:
            st.write("**⚠️ Recommendations:**")
            if dataset_info['total_images'] < 100:
                st.write("- Tambahkan lebih banyak data")
            if dataset_info['benign_count'] == 0 or dataset_info['malicious_count'] == 0:
                st.write("- Tambahkan gambar untuk kelas yang kosong")
            if dataset_info['total_images'] > 0 and abs(dataset_info['benign_count'] - dataset_info['malicious_count']) / dataset_info['total_images'] > 0.5:
                st.write("- Seimbangkan distribusi kelas")
            else : 
                st.write("Dataset sudah sangat baik!")

class ModelResultsAnalyzer:
    """Class untuk analisis dan visualisasi hasil model"""
    
    def __init__(self):
        self.model_results = {}
        self.model_histories = {}
        self.best_model = None
    
    def load_model_results(self, model_name, eval_file, history_file=None):
        """Load hasil model dari file JSON"""
        try:
            # Load evaluation results
            eval_content = eval_file.getvalue().decode('utf-8') if hasattr(eval_file, 'getvalue') else eval_file.read().decode('utf-8')
            eval_data = json.loads(eval_content)
            
            # Validasi format file
            required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            for metric in required_metrics:
                if metric not in eval_data:
                    st.error(f"❌ File {model_name}_eval.json tidak memiliki metric '{metric}'")
                    return False
            
            # Load history jika ada
            history_data = {}
            if history_file:
                try:
                    history_content = history_file.getvalue().decode('utf-8') if hasattr(history_file, 'getvalue') else history_file.read().decode('utf-8')
                    history_data = json.loads(history_content)
                except:
                    st.warning(f"⚠️ History untuk {model_name} tidak valid atau kosong")
            
            # Validasi metrics tidak semua 1.0
            if self._check_suspicious_metrics(eval_data):
                st.warning(f"⚠️ Metrics untuk {model_name} mencurigakan (semua 1.0 atau 0.0)")
            
            # Simpan hasil
            self.model_results[model_name] = eval_data
            if history_data:
                self.model_histories[model_name] = history_data
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading {model_name} results: {str(e)}")
            return False
    
    def _check_suspicious_metrics(self, metrics):
        """Cek apakah metrics mencurigakan (semua 1.0 atau 0.0)"""
        if not metrics:
            return False
        
        metric_keys = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [metrics.get(k, 0) for k in metric_keys]
        
        # Cek jika semua 1.0
        all_ones = all(abs(v - 1.0) < 0.001 for v in values)
        # Cek jika semua 0.0
        all_zeros = all(abs(v - 0.0) < 0.001 for v in values)
        
        return all_ones or all_zeros
    
    def get_metric_color(self, value):
        """Get color based on metric value"""
        if value >= 0.9:
            return '#10b981'  # Green
        elif value >= 0.8:
            return '#f59e0b'  # Yellow
        elif value >= 0.7:
            return '#ef4444'  # Red
        else:
            return '#6b7280'  # Gray
    
    def display_model_card(self, model_name, model_results):
        """Display model results card"""
        card_class = f"{model_name}-card"
        
        st.markdown(f'<div class="model-card {card_class}">', unsafe_allow_html=True)
        
        # Header dengan icon
        icons = {
            'cnn': '🖼️',
            'lstm': '📝', 
            'gru': '🌀'
        }
        
        col_header = st.columns([1, 8])
        with col_header[0]:
            st.markdown(f"<h2 style='margin:0;'>{icons.get(model_name, '🤖')}</h2>", unsafe_allow_html=True)
        with col_header[1]:
            st.subheader(f"{model_name.upper()} Results")
        
        # Metrics dalam 4 kolom
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            accuracy = model_results.get('accuracy', 0)
            color = self.get_metric_color(accuracy)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value" style="color: {color}">
                    {accuracy:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            precision = model_results.get('precision', 0)
            color = self.get_metric_color(precision)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value" style="color: {color}">
                    {precision:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            recall = model_results.get('recall', 0)
            color = self.get_metric_color(recall)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value" style="color: {color}">
                    {recall:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            f1_score = model_results.get('f1_score', 0)
            color = self.get_metric_color(f1_score)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value" style="color: {color}">
                    {f1_score:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional metrics jika ada
        additional_metrics = {k: v for k, v in model_results.items() 
                            if k not in ['accuracy', 'precision', 'recall', 'f1_score'] 
                            and isinstance(v, (int, float))}
        
        if additional_metrics:
            st.write("**Additional Metrics:**")
            cols = st.columns(len(additional_metrics))
            for idx, (metric_name, value) in enumerate(additional_metrics.items()):
                with cols[idx]:
                    st.metric(metric_name.replace('_', ' ').title(), f"{value:.4f}")
        
        # Plot training history jika ada
        if model_name in self.model_histories:
            self.plot_training_history(model_name)
        
        # Warning jika metrics mencurigakan
        if self._check_suspicious_metrics(model_results):
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.warning("⚠️ **Warning:** Metrics menunjukkan kemungkinan overfitting atau data leakage (semua metrics = 1.0)")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def plot_training_history(self, model_name):
        """Plot training history menggunakan matplotlib"""
        if model_name not in self.model_histories:
            return
        
        history = self.model_histories[model_name]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1 = axes[0]
        if 'accuracy' in history:
            ax1.plot(history['accuracy'], label='Training', linewidth=2, color='#3B82F6')
        if 'val_accuracy' in history:
            ax1.plot(history['val_accuracy'], label='Validation', linewidth=2, color='#10B981')
        ax1.set_title(f'{model_name.upper()} Accuracy History', fontsize=14, fontweight='bold')
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
        ax2.set_title(f'{model_name.upper()} Loss History', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def determine_best_model(self):
        """Tentukan model terbaik berdasarkan F1-Score"""
        if not self.model_results:
            return None, None
        
        best_model = None
        best_f1 = -1
        
        for model_name, results in self.model_results.items():
            f1 = results.get('f1_score', 0)
            if f1 > best_f1:
                best_f1 = f1
                best_model = model_name
        
        self.best_model = best_model
        return best_model, best_f1
    
    def create_comparison_table(self):
        """Buat tabel perbandingan semua model"""
        if not self.model_results:
            return None
        
        comparison_data = []
        for model_name, results in self.model_results.items():
            row = {
                'Model': model_name.upper(),
                'Accuracy': results.get('accuracy', 0),
                'Precision': results.get('precision', 0),
                'Recall': results.get('recall', 0),
                'F1-Score': results.get('f1_score', 0)
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def plot_comparison_chart(self):
        """Plot chart perbandingan model"""
        if not self.model_results:
            return
        
        # Prepare data
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        models = list(self.model_results.keys())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Colors for each model
        colors = {
            'cnn': '#3B82F6',
            'lstm': '#10B981',
            'gru': '#8B5CF6'
        }
        
        bar_width = 0.2
        x = np.arange(len(metrics))
        
        # Add bars for each model
        for i, model in enumerate(models):
            values = [
                self.model_results[model].get('accuracy', 0),
                self.model_results[model].get('precision', 0),
                self.model_results[model].get('recall', 0),
                self.model_results[model].get('f1_score', 0)
            ]
            
            ax.bar(x + i * bar_width, values, bar_width, 
                  label=model.upper(), color=colors.get(model, '#94a3b8'))
            
            # Add value labels on bars
            for j, v in enumerate(values):
                ax.text(x[j] + i * bar_width, v + 0.01, f'{v:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + bar_width * (len(models) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add threshold lines
        ax.axhline(y=0.9, color='#10b981', linestyle='--', alpha=0.3, label='Excellent')
        ax.axhline(y=0.8, color='#f59e0b', linestyle='--', alpha=0.3, label='Good')
        ax.axhline(y=0.7, color='#ef4444', linestyle='--', alpha=0.3, label='Fair')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def radar_chart(self):
        """Create radar chart untuk perbandingan model"""
        if len(self.model_results) < 2:
            return
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Colors for each model
        colors = {
            'cnn': '#3B82F6',
            'lstm': '#10B981',
            'gru': '#8B5CF6'
        }
        
        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        # Plot each model
        for model_name, results in self.model_results.items():
            values = [
                results.get('accuracy', 0),
                results.get('precision', 0),
                results.get('recall', 0),
                results.get('f1_score', 0)
            ]
            values += values[:1]  # Close the polygon
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name.upper(), 
                   color=colors.get(model_name, '#94a3b8'))
            ax.fill(angles, values, alpha=0.1, color=colors.get(model_name, '#94a3b8'))
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim([0, 1])
        ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)

def main():
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">🤖 QR Code Dataset & Model Results</h1>
    </div>
    """, unsafe_allow_html=True)
    
    dataset_preview = DatasetPreview()
    analyzer = ModelResultsAnalyzer()
    
    with st.sidebar:
        st.header("📤 Upload Files")
        
        # Tab untuk upload
        tab_upload, tab_info = st.tabs(["📁 Upload", "ℹ️ Info"])
        
        with tab_upload:           
            # Upload Dataset
            st.subheader("📂 Dataset ZIP")
            uploaded_dataset = st.file_uploader(
                "Upload dataset ZIP for preview",
                type=['zip'],
                help="Must contain 'benign' and 'malicious' folders. For preview only.",
                key="dataset_zip"
            )
            
            st.markdown("---")
            
            # Upload Model Results
            st.subheader("🤖  Model Results")
            
            st.markdown("🖼️ **CNN Results**")
            cnn_eval = st.file_uploader(
                "Upload cnn_eval.json",
                type=["json"],
                key="cnn_eval"
            )
            cnn_history = st.file_uploader(
                "Upload cnn_history.json (optional)",
                type=["json"],
                key="cnn_history"
            )
            
            st.markdown("📝 **LSTM Results**")
            lstm_eval = st.file_uploader(
                "Upload lstm_eval.json",
                type=["json"],
                key="lstm_eval"
            )
            lstm_history = st.file_uploader(
                "Upload lstm_history.json (optional)",
                type=["json"],
                key="lstm_history"
            )
            
            st.markdown("🌀 **GRU Results**")
            gru_eval = st.file_uploader(
                "Upload gru_eval.json",
                type=["json"],
                key="gru_eval"
            )
            gru_history = st.file_uploader(
                "Upload gru_history.json (optional)",
                type=["json"],
                key="gru_history"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Load buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📂 Load Dataset", use_container_width=True):
                    if uploaded_dataset:
                        with st.spinner("Processing dataset..."):
                            dataset_info = dataset_preview.process_zip_file(uploaded_dataset)
                            if dataset_info:
                                st.session_state['dataset_info'] = dataset_info
                                st.success(f"✅ Loaded {dataset_info['total_images']} images")
                            else:
                                st.error("❌ Failed to load dataset")
                    else:
                        st.warning("Please upload dataset ZIP first")
            
            with col2:
                if st.button("🤖 Load Model Results", type="primary", use_container_width=True):
                    # Reset previous results
                    analyzer.model_results = {}
                    analyzer.model_histories = {}
                    
                    models_loaded = 0
                    
                    # Load CNN results
                    if cnn_eval:
                        with st.spinner("Loading CNN results..."):
                            if analyzer.load_model_results('cnn', cnn_eval, cnn_history):
                                models_loaded += 1
                    
                    # Load LSTM results
                    if lstm_eval:
                        with st.spinner("Loading LSTM results..."):
                            if analyzer.load_model_results('lstm', lstm_eval, lstm_history):
                                models_loaded += 1
                    
                    # Load GRU results
                    if gru_eval:
                        with st.spinner("Loading GRU results..."):
                            if analyzer.load_model_results('gru', gru_eval, gru_history):
                                models_loaded += 1
                    
                    if models_loaded > 0:
                        st.session_state['analyzer'] = analyzer
                        st.session_state['models_loaded'] = models_loaded
                        st.success(f"✅ Loaded {models_loaded} model(s)")
                    else:
                        st.error("❌ No valid model results loaded")
            
            st.markdown("---")
            
            if st.button("🔄 Reset All", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        
        with tab_info:
            st.markdown("### 📋 File Formats")
            
            st.markdown("**Dataset ZIP:**")
            st.code("""
            dataset.zip/
            ├── benign/
            │   ├── qr1.png
            │   ├── qr2.jpg
            │   ├── qr3.jpeg
            │   └── ...
            └── malicious/
                ├── malware1.png
                ├── malware2.jpg
                ├── malware3.jpeg
                └── ...
            """)
            
            st.markdown("**Model Eval JSON:**")
            st.code("""
            {
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.93,
                "f1_score": 0.935,
                "auc": 0.98  // optional
            }
            """)
            
            st.markdown("**Model History JSON (optional):**")
            st.code("""
            {
                "accuracy": [0.85, 0.90, 0.92, 0.94, 0.95],
                "val_accuracy": [0.82, 0.87, 0.90, 0.92, 0.93],
                "loss": [0.45, 0.30, 0.22, 0.18, 0.15],
                "val_loss": [0.48, 0.32, 0.25, 0.20, 0.17]
            }
            """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📂 Dataset", "📊 Model Results", "📈 Comparison", "💾 Download"])
    
    # --- TAB 1: Dataset Preview ---
    with tab1:
        st.header("📂 Dataset Preview")
        
        if 'dataset_info' in st.session_state:
            dataset_info = st.session_state['dataset_info']
            dataset_preview.display_dataset_info(dataset_info)
            
            # Cleanup button
            if st.button("🗑️ Clear Dataset Preview", type="secondary"):
                if 'temp_dir' in dataset_info:
                    try:
                        shutil.rmtree(dataset_info['temp_dir'], ignore_errors=True)
                    except:
                        pass
                del st.session_state['dataset_info']
                st.rerun()
        else:
            st.info("👈 Upload dataset ZIP in the sidebar to preview")
            
            # Create a placeholder without external image
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                        border-radius: 10px; 
                        padding: 3rem; 
                        text-align: center;
                        border: 2px dashed #cbd5e1;">
                <h3 style="color: #3b82f6;">📂 Dataset Preview Area</h3>
                <p style="color: #64748b;">Upload a dataset ZIP file to see preview here</p>
                <p style="font-size: 48px;">📊</p>
            </div>
            """, unsafe_allow_html=True)
    
    # --- TAB 2: Model Results ---
    with tab2:
        st.header("📊 Individual Model Results")
        
        if st.session_state.get('models_loaded') and 'analyzer' in st.session_state:
            analyzer = st.session_state['analyzer']
            
            # Display loaded models
            st.write(f"**ℹ️ Loaded Models:** {', '.join([m.upper() for m in analyzer.model_results.keys()])}")
            
            # Display each model's results
            if 'cnn' in analyzer.model_results:
                analyzer.display_model_card('cnn', analyzer.model_results['cnn'])
            
            if 'lstm' in analyzer.model_results:
                analyzer.display_model_card('lstm', analyzer.model_results['lstm'])
            
            if 'gru' in analyzer.model_results:
                analyzer.display_model_card('gru', analyzer.model_results['gru'])
            
            # Summary statistics
            if len(analyzer.model_results) > 0:
                st.markdown("### 📈 Summary Statistics")
                
                summary_data = []
                for model_name, results in analyzer.model_results.items():
                    summary_data.append({
                        'Model': model_name.upper(),
                        'Accuracy': f"{results.get('accuracy', 0):.4f}",
                        'Precision': f"{results.get('precision', 0):.4f}",
                        'Recall': f"{results.get('recall', 0):.4f}",
                        'F1-Score': f"{results.get('f1_score', 0):.4f}"
                    })
                
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, width=800)
                
                # Performance analysis
                st.markdown("### 🎯 Performance Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Best accuracy
                    best_acc_model = max(analyzer.model_results.items(), 
                                       key=lambda x: x[1].get('accuracy', 0))
                    st.metric("Best Accuracy", 
                             f"{best_acc_model[1].get('accuracy', 0):.4f}",
                             best_acc_model[0].upper())
                
                with col2:
                    # Best F1-score
                    best_f1_model = max(analyzer.model_results.items(), 
                                      key=lambda x: x[1].get('f1_score', 0))
                    st.metric("Best F1-Score", 
                             f"{best_f1_model[1].get('f1_score', 0):.4f}",
                             best_f1_model[0].upper())
                
                with col3:
                    # Average F1-score
                    avg_f1 = np.mean([r.get('f1_score', 0) for r in analyzer.model_results.values()])
                    st.metric("Average F1-Score", f"{avg_f1:.4f}")
        else:
            st.info("👈 Upload model results in the sidebar and click 'Load Model Results'")
            
            # Create a placeholder without external image
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); 
                        border-radius: 10px; 
                        padding: 3rem; 
                        text-align: center;
                        border: 2px dashed #cbd5e1;">
                <h3 style="color: #10b981;">🤖 Model Results Area</h3>
                <p style="color: #64748b;">Upload model results to see analysis here</p>
                <p style="font-size: 48px;">📈</p>
            </div>
            """, unsafe_allow_html=True)
    
    # --- TAB 3: Comparison ---
    with tab3:
        st.header("📈 Model Comparison Analysis")
        
        if st.session_state.get('models_loaded') and 'analyzer' in st.session_state:
            analyzer = st.session_state['analyzer']
            
            if len(analyzer.model_results) >= 2:
                # Comparison table
                st.subheader("📊 Performance Comparison Table")
                comparison_df = analyzer.create_comparison_table()
                if comparison_df is not None:
                    # Style the dataframe
                    def highlight_cells(val):
                        if isinstance(val, (int, float)):
                            if val >= 0.9:
                                return 'background-color: #dcfce7; color: #166534;'
                            elif val >= 0.8:
                                return 'background-color: #fef9c3; color: #854d0e;'
                            elif val >= 0.7:
                                return 'background-color: #fee2e2; color: #991b1b;'
                        return ''
                    
                    styled_df = comparison_df.style.applymap(highlight_cells, 
                                                           subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
                    st.dataframe(styled_df, width=800)
                
                # Bar chart comparison
                st.subheader("📊 Bar Chart Comparison")
                analyzer.plot_comparison_chart()
                
                # Radar chart
                st.subheader("🎯 Radar Chart Comparison")
                analyzer.radar_chart()
                
                # Best model determination
                st.subheader("🏆 Best Model Analysis")
                best_model_name, best_f1 = analyzer.determine_best_model()
                
                if best_model_name:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Best Model", best_model_name.upper())
                    
                    with col2:
                        best_results = analyzer.model_results[best_model_name]
                        st.metric("Best F1-Score", f"{best_f1:.4f}")
                    
                    with col3:
                        improvement = 0
                        if len(analyzer.model_results) > 1:
                            other_f1 = [r.get('f1_score', 0) for n, r in analyzer.model_results.items() 
                                      if n != best_model_name]
                            if other_f1 and np.mean(other_f1) > 0:
                                avg_other_f1 = np.mean(other_f1)
                                improvement = ((best_f1 - avg_other_f1) / avg_other_f1) * 100
                                st.metric("Improvement", f"{improvement:.1f}%")
                    
                    # Recommendations
                    st.markdown("### 💡 Recommendations")
                    
                    if best_f1 >= 0.9:
                        st.write("✅ **Excellent performance!** Model sudah sangat baik.")
                    elif best_f1 >= 0.8:
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.write("ℹ️ **Good performance.** Consider fine-tuning for better results.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.write("⚠️ **Performance needs improvement.** Consider:")
                        st.write("- Collecting more training data")
                        st.write("- Trying different model architectures")
                        st.write("- Hyperparameter tuning")
                        st.markdown('</div>', unsafe_allow_html=True)
                
            else:
                st.warning("⚠️ Need at least 2 models for comparison")
                if len(analyzer.model_results) == 1:
                    model_name = list(analyzer.model_results.keys())[0]
                    st.info(f"Only {model_name.upper()} loaded. Upload more models for comparison.")
        else:
            st.info("👈 Upload at least 2 model results for comparison")
    
    # --- TAB 4: Download ---
    with tab4:
        st.header("💾 Download Reports")
        
        if st.session_state.get('models_loaded') and 'analyzer' in st.session_state:
            analyzer = st.session_state['analyzer']
            
            if len(analyzer.model_results) > 0:
                # Create comprehensive report
                st.subheader("📄 Generate Report")
                
                report_format = st.selectbox(
                    "Select Report Format:",
                    ["JSON", "CSV", "Text Summary"]
                )
                
                if st.button("📥 Generate Comprehensive Report", use_container_width=True):
                    with st.spinner("Generating report..."):
                        # Create report data
                        report_data = {
                            'analysis_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'models_analyzed': list(analyzer.model_results.keys()),
                            'model_results': analyzer.model_results,
                            'best_model': analyzer.determine_best_model()[0],
                            'best_f1_score': analyzer.determine_best_model()[1],
                            'summary_statistics': {}
                        }
                        
                        # Add summary statistics
                        if len(analyzer.model_results) > 0:
                            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                                values = [r.get(metric, 0) for r in analyzer.model_results.values()]
                                report_data['summary_statistics'][metric] = {
                                    'mean': np.mean(values),
                                    'std': np.std(values),
                                    'min': np.min(values),
                                    'max': np.max(values),
                                    'median': np.median(values)
                                }
                        
                        if report_format == "JSON":
                            json_data = json.dumps(report_data, indent=2)
                            st.download_button(
                                label="📥 Download JSON Report",
                                data=json_data,
                                file_name=f"model_analysis_{time.strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        
                        elif report_format == "CSV":
                            # Create comparison CSV
                            comparison_df = analyzer.create_comparison_table()
                            csv_data = comparison_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download CSV Report",
                                data=csv_data,
                                file_name=f"model_comparison_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        elif report_format == "Text Summary":
                            # Create text summary
                            summary_text = f"""
                            MODEL ANALYSIS REPORT
                            ======================
                            
                            Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
                            
                            Models Analyzed: {', '.join(analyzer.model_results.keys())}
                            
                            Best Model: {analyzer.determine_best_model()[0].upper()}
                            Best F1-Score: {analyzer.determine_best_model()[1]:.4f}
                            
                            {'='*60}
                            
                            DETAILED RESULTS:
                            {'='*60}
                            """
                            
                            for model_name, results in analyzer.model_results.items():
                                summary_text += f"""
                            {model_name.upper()}:
                              Accuracy:  {results.get('accuracy', 0):.4f}
                              Precision: {results.get('precision', 0):.4f}
                              Recall:    {results.get('recall', 0):.4f}
                              F1-Score:  {results.get('f1_score', 0):.4f}
                            
                            """
                            
                            st.download_button(
                                label="📥 Download Text Summary",
                                data=summary_text,
                                file_name=f"model_summary_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                
                # Individual model exports
                st.subheader("📊 Export Individual Model Results")
                
                col1, col2, col3 = st.columns(3)
                
                export_models = []
                if 'cnn' in analyzer.model_results:
                    export_models.append(('cnn', '🖼️ CNN'))
                if 'lstm' in analyzer.model_results:
                    export_models.append(('lstm', '📝 LSTM'))
                if 'gru' in analyzer.model_results:
                    export_models.append(('gru', '🌀 GRU'))
                
                for idx, (model_name, display_name) in enumerate(export_models):
                    with [col1, col2, col3][idx % 3]:
                        results = analyzer.model_results[model_name]
                        json_str = json.dumps(results, indent=2)
                        
                        st.download_button(
                            label=f"Download {display_name}",
                            data=json_str,
                            file_name=f"{model_name}_results.json",
                            mime="application/json"
                        )
                
                # Quick summary
                st.subheader("📋 Quick Summary")
                
                if 'dataset_info' in st.session_state:
                    dataset_info = st.session_state['dataset_info']
                    dataset_summary = f"""
                    Dataset: {dataset_info['total_images']} images
                    - Benign: {dataset_info['benign_count']}
                    - Malicious: {dataset_info['malicious_count']}
                    """
                else:
                    dataset_summary = "No dataset loaded"
                
                summary_text = f"""
                QUICK MODEL ANALYSIS SUMMARY
                {'='*40}
                
                Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
                
                DATASET INFO:
                {dataset_summary}
                
                MODEL RESULTS:
                {'='*40}
                
                """
                
                for model_name, results in analyzer.model_results.items():
                    summary_text += f"""
                {model_name.upper()}:
                  Accuracy:  {results.get('accuracy', 0):.4f}
                  Precision: {results.get('precision', 0):.4f}
                  Recall:    {results.get('recall', 0):.4f}
                  F1-Score:  {results.get('f1_score', 0):.4f}
                
                """
                
                best_model, best_f1 = analyzer.determine_best_model()
                summary_text += f"""
                CONCLUSION:
                {'='*40}
                
                Best Model: {best_model.upper()}
                Best F1-Score: {best_f1:.4f}
                
                Recommendation: {'Excellent' if best_f1 >= 0.9 else 'Good' if best_f1 >= 0.8 else 'Needs Improvement'}
                """
                
                st.download_button(
                    label="📝 Download Quick Summary",
                    data=summary_text,
                    file_name=f"quick_summary_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
            else:
                st.info("No model results to export")
        else:
            st.info("👈 Load model results first to generate reports")

if __name__ == "__main__":
    main()