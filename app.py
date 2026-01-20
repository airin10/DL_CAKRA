"""
QR Code Security Analyzer - MODEL RESULTS DISPLAY ONLY
[Hanya upload dan tampilkan hasil CNN, LSTM, GRU tanpa training]
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# === PAGE CONFIG ===
st.set_page_config(
    page_title="QR Code Model Results Viewer",
    page_icon="📊",
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
</style>
""", unsafe_allow_html=True)

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
            
            # Simpan hasil
            self.model_results[model_name] = eval_data
            if history_data:
                self.model_histories[model_name] = history_data
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading {model_name} results: {str(e)}")
            return False
    
    def validate_metrics(self, metrics):
        """Validasi apakah metrics valid (tidak semua 1.0)"""
        if not metrics:
            return False
        
        # Cek jika semua metrics 1.0 atau 0.0
        all_ones = all(abs(v - 1.0) < 0.001 for k, v in metrics.items() 
                      if k in ['accuracy', 'precision', 'recall', 'f1_score'])
        all_zeros = all(abs(v - 0.0) < 0.001 for k, v in metrics.items() 
                       if k in ['accuracy', 'precision', 'recall', 'f1_score'])
        
        if all_ones:
            st.warning("⚠️ Semua metrics = 1.0 (kemungkinan overfitting atau data leakage)")
            return False
        elif all_zeros:
            st.warning("⚠️ Semua metrics = 0.0 (model tidak belajar)")
            return False
        
        return True
    
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
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def plot_training_history(self, model_name):
        """Plot training history menggunakan Plotly"""
        if model_name not in self.model_histories:
            return
        
        history = self.model_histories[model_name]
        
        # Create subplots
        fig = make_subplots(
            rows=1, 
            cols=2,
            subplot_titles=(f'{model_name.upper()} Accuracy History', 
                           f'{model_name.upper()} Loss History')
        )
        
        # Accuracy plot
        if 'accuracy' in history:
            fig.add_trace(
                go.Scatter(
                    y=history['accuracy'],
                    mode='lines+markers',
                    name='Training Accuracy',
                    line=dict(color='#3B82F6', width=2),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
        
        if 'val_accuracy' in history:
            fig.add_trace(
                go.Scatter(
                    y=history['val_accuracy'],
                    mode='lines+markers',
                    name='Validation Accuracy',
                    line=dict(color='#10B981', width=2),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
        
        # Loss plot
        if 'loss' in history:
            fig.add_trace(
                go.Scatter(
                    y=history['loss'],
                    mode='lines+markers',
                    name='Training Loss',
                    line=dict(color='#EF4444', width=2),
                    marker=dict(size=6)
                ),
                row=1, col=2
            )
        
        if 'val_loss' in history:
            fig.add_trace(
                go.Scatter(
                    y=history['val_loss'],
                    mode='lines+markers',
                    name='Validation Loss',
                    line=dict(color='#F59E0B', width=2),
                    marker=dict(size=6)
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1, range=[0, 1])
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
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
            return None
        
        # Prepare data
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        models = list(self.model_results.keys())
        
        fig = go.Figure()
        
        # Colors for each model
        colors = {
            'cnn': '#3B82F6',
            'lstm': '#10B981',
            'gru': '#8B5CF6'
        }
        
        # Add bars for each model
        for model in models:
            values = [
                self.model_results[model].get('accuracy', 0),
                self.model_results[model].get('precision', 0),
                self.model_results[model].get('recall', 0),
                self.model_results[model].get('f1_score', 0)
            ]
            
            fig.add_trace(go.Bar(
                name=model.upper(),
                x=metrics,
                y=values,
                marker_color=colors.get(model, '#94a3b8'),
                text=[f'{v:.3f}' for v in values],
                textposition='auto',
            ))
        
        # Update layout
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Metrics",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            barmode='group',
            template="plotly_white",
            height=500,
            showlegend=True
        )
        
        # Add threshold lines
        fig.add_hline(y=0.9, line_dash="dash", line_color="#10b981", 
                     annotation_text="Excellent", annotation_position="right")
        fig.add_hline(y=0.8, line_dash="dash", line_color="#f59e0b", 
                     annotation_text="Good", annotation_position="right")
        fig.add_hline(y=0.7, line_dash="dash", line_color="#ef4444", 
                     annotation_text="Fair", annotation_position="right")
        
        return fig
    
    def radar_chart(self):
        """Create radar chart untuk perbandingan model"""
        if len(self.model_results) < 2:
            return None
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig = go.Figure()
        
        # Colors for each model
        colors = {
            'cnn': '#3B82F6',
            'lstm': '#10B981',
            'gru': '#8B5CF6'
        }
        
        for model_name, results in self.model_results.items():
            values = [
                results.get('accuracy', 0),
                results.get('precision', 0),
                results.get('recall', 0),
                results.get('f1_score', 0)
            ]
            
            # Add model trace
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=metrics + [metrics[0]],
                name=model_name.upper(),
                fill='toself',
                line_color=colors.get(model_name, '#94a3b8'),
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart",
            height=500,
            template="plotly_white"
        )
        
        return fig

def main():
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">📊 QR Code Model Results Viewer</h1>
        <p class="hero-subtitle">
            Upload and visualize CNN, LSTM, and GRU model results without training
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    analyzer = ModelResultsAnalyzer()
    
    with st.sidebar:
        st.header("📤 Upload Model Results")
        
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        # Upload untuk setiap model
        st.subheader("🖼️ CNN Results")
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
        
        st.markdown("---")
        st.subheader("📝 LSTM Results")
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
        
        st.markdown("---")
        st.subheader("🌀 GRU Results")
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
        st.header("⚙️ Settings")
        
        show_history = st.checkbox("Show Training History", value=True,
                                  help="Display training history plots if available")
        
        auto_validate = st.checkbox("Auto-Validate Metrics", value=True,
                                   help="Check for suspicious metrics (all 1.0 or 0.0)")
        
        if st.button("🚀 Load & Analyze Results", type="primary", use_container_width=True):
            # Reset previous results
            analyzer.model_results = {}
            analyzer.model_histories = {}
            
            # Load CNN results
            if cnn_eval:
                with st.spinner("Loading CNN results..."):
                    if analyzer.load_model_results('cnn', cnn_eval, cnn_history):
                        st.session_state['cnn_loaded'] = True
                        if auto_validate:
                            analyzer.validate_metrics(analyzer.model_results.get('cnn', {}))
            
            # Load LSTM results
            if lstm_eval:
                with st.spinner("Loading LSTM results..."):
                    if analyzer.load_model_results('lstm', lstm_eval, lstm_history):
                        st.session_state['lstm_loaded'] = True
                        if auto_validate:
                            analyzer.validate_metrics(analyzer.model_results.get('lstm', {}))
            
            # Load GRU results
            if gru_eval:
                with st.spinner("Loading GRU results..."):
                    if analyzer.load_model_results('gru', gru_eval, gru_history):
                        st.session_state['gru_loaded'] = True
                        if auto_validate:
                            analyzer.validate_metrics(analyzer.model_results.get('gru', {}))
            
            if analyzer.model_results:
                st.session_state['analyzer'] = analyzer
                st.session_state['results_loaded'] = True
                st.success(f"✅ Loaded {len(analyzer.model_results)} model(s)")
            else:
                st.error("❌ No valid model results loaded")
        
        st.markdown("---")
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        
        st.markdown("---")
        st.header("ℹ️ How to Use")
        with st.expander("Instructions"):
            st.markdown("""
            1. **Upload JSON Files** for each model:
               - `*_eval.json`: Contains accuracy, precision, recall, f1_score
               - `*_history.json` (optional): Contains training history
            
            2. **Click 'Load & Analyze Results'**
            
            3. **View results** in the main tabs
            
            **Required format for *_eval.json:**
            ```json
            {
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.93,
                "f1_score": 0.935
            }
            ```
            
            **Optional format for *_history.json:**
            ```json
            {
                "accuracy": [0.85, 0.90, 0.92, 0.94, 0.95],
                "val_accuracy": [0.82, 0.87, 0.90, 0.92, 0.93],
                "loss": [0.45, 0.30, 0.22, 0.18, 0.15],
                "val_loss": [0.48, 0.32, 0.25, 0.20, 0.17]
            }
            ```
            """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Results", "📈 Comparison", "🏆 Best Model", "💾 Download"])
    
    # --- TAB 1: Model Results ---
    with tab1:
        st.header("📊 Individual Model Results")
        
        if st.session_state.get('results_loaded') and 'analyzer' in st.session_state:
            analyzer = st.session_state['analyzer']
            
            # Display loaded models
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.write(f"**Loaded Models:** {', '.join([m.upper() for m in analyzer.model_results.keys()])}")
            st.markdown('</div>', unsafe_allow_html=True)
            
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
                st.dataframe(df_summary, use_container_width=True)
                
        else:
            st.info("👈 Please upload model results in the sidebar and click 'Load & Analyze Results'")
    
    # --- TAB 2: Comparison ---
    with tab2:
        st.header("📈 Model Comparison Analysis")
        
        if st.session_state.get('results_loaded') and 'analyzer' in st.session_state:
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
                    st.dataframe(styled_df, use_container_width=True)
                
                # Bar chart comparison
                st.subheader("📊 Bar Chart Comparison")
                fig_bar = analyzer.plot_comparison_chart()
                if fig_bar:
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Radar chart
                st.subheader("🎯 Radar Chart Comparison")
                fig_radar = analyzer.radar_chart()
                if fig_radar:
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                # Statistical analysis
                st.subheader("📊 Statistical Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if len(analyzer.model_results) > 0:
                        avg_accuracy = np.mean([r.get('accuracy', 0) for r in analyzer.model_results.values()])
                        st.metric("Average Accuracy", f"{avg_accuracy:.4f}")
                
                with col2:
                    if len(analyzer.model_results) > 0:
                        avg_f1 = np.mean([r.get('f1_score', 0) for r in analyzer.model_results.values()])
                        st.metric("Average F1-Score", f"{avg_f1:.4f}")
                
                with col3:
                    if len(analyzer.model_results) > 0:
                        std_accuracy = np.std([r.get('accuracy', 0) for r in analyzer.model_results.values()])
                        st.metric("Std Dev Accuracy", f"{std_accuracy:.4f}")
                
            else:
                st.warning("⚠️ Need at least 2 models for comparison")
                if len(analyzer.model_results) == 1:
                    model_name = list(analyzer.model_results.keys())[0]
                    st.info(f"Only {model_name.upper()} loaded. Upload more models for comparison.")
        else:
            st.info("👈 Please upload at least 2 model results for comparison")
    
    # --- TAB 3: Best Model ---
    with tab3:
        st.header("🏆 Best Model Analysis")
        
        if st.session_state.get('results_loaded') and 'analyzer' in st.session_state:
            analyzer = st.session_state['analyzer']
            
            if len(analyzer.model_results) > 0:
                # Determine best model
                best_model_name, best_f1 = analyzer.determine_best_model()
                
                if best_model_name:
                    best_results = analyzer.model_results[best_model_name]
                    
                    # Best model banner
                    st.markdown(f"""
                    <div class="best-model-banner">
                        <h3 style="margin:0 0 1rem 0;">🏆 Best Performing Model: {best_model_name.upper()}</h3>
                        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
                            <div style="text-align: center;">
                                <div style="font-size: 0.9rem; color: #64748b;">F1-Score</div>
                                <div style="font-size: 2rem; font-weight: 700; color: #10b981;">
                                    {best_f1:.4f}
                                </div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 0.9rem; color: #64748b;">Accuracy</div>
                                <div style="font-size: 2rem; font-weight: 700; color: #10b981;">
                                    {best_results.get('accuracy', 0):.4f}
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Best model details
                    st.subheader(f"📊 {best_model_name.upper()} Detailed Metrics")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Metrics gauge chart
                        fig = go.Figure()
                        
                        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                        values = [
                            best_results.get('accuracy', 0),
                            best_results.get('precision', 0),
                            best_results.get('recall', 0),
                            best_results.get('f1_score', 0)
                        ]
                        
                        colors = ['#10b981' if v >= 0.8 else '#f59e0b' if v >= 0.7 else '#ef4444' for v in values]
                        
                        fig.add_trace(go.Bar(
                            x=values,
                            y=metrics,
                            orientation='h',
                            marker_color=colors,
                            text=[f'{v:.3f}' for v in values],
                            textposition='auto',
                        ))
                        
                        fig.update_layout(
                            title=f"{best_model_name.upper()} Performance Metrics",
                            xaxis=dict(range=[0, 1]),
                            height=300,
                            showlegend=False,
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Quick stats
                        st.markdown("### 📈 Quick Stats")
                        
                        st.metric("Precision", f"{best_results.get('precision', 0):.4f}")
                        st.metric("Recall", f"{best_results.get('recall', 0):.4f}")
                        
                        # Calculate improvement if multiple models
                        if len(analyzer.model_results) > 1:
                            other_f1 = [r.get('f1_score', 0) for n, r in analyzer.model_results.items() 
                                      if n != best_model_name]
                            if other_f1:
                                avg_other_f1 = np.mean(other_f1)
                                improvement = ((best_f1 - avg_other_f1) / avg_other_f1) * 100
                                st.metric("Improvement vs Others", f"{improvement:.1f}%")
                    
                    # Model recommendations
                    st.subheader("💡 Recommendations")
                    
                    recommendations = []
                    if best_results.get('precision', 0) < 0.8:
                        recommendations.append("Improve precision to reduce false positives")
                    if best_results.get('recall', 0) < 0.8:
                        recommendations.append("Improve recall to reduce false negatives")
                    if best_f1 < 0.8:
                        recommendations.append("Overall model performance needs improvement")
                    
                    if recommendations:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.write("**Areas for Improvement:**")
                        for rec in recommendations:
                            st.write(f"- {rec}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.write("✅ Excellent model performance across all metrics!")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
            else:
                st.info("No model results available")
        else:
            st.info("👈 Please load model results first")
    
    # --- TAB 4: Download ---
    with tab4:
        st.header("💾 Download Analysis Results")
        
        if st.session_state.get('results_loaded') and 'analyzer' in st.session_state:
            analyzer = st.session_state['analyzer']
            
            if len(analyzer.model_results) > 0:
                # Create comprehensive report
                st.subheader("📄 Generate Report")
                
                report_format = st.selectbox(
                    "Select Report Format:",
                    ["JSON", "CSV", "HTML", "PDF (Not supported)"]
                )
                
                if st.button("📥 Generate Comprehensive Report", use_container_width=True):
                    with st.spinner("Generating report..."):
                        # Create report data
                        report_data = {
                            'analysis_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'models_analyzed': list(analyzer.model_results.keys()),
                            'model_results': analyzer.model_results,
                            'best_model': analyzer.determine_best_model()[0],
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
                                    'max': np.max(values)
                                }
                        
                        if report_format == "JSON":
                            json_data = json.dumps(report_data, indent=2)
                            st.download_button(
                                label="📥 Download JSON Report",
                                data=json_data,
                                file_name=f"model_analysis_{time.strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                        elif report_format == "CSV":
                            # Create comparison CSV
                            comparison_df = analyzer.create_comparison_table()
                            csv_data = comparison_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download CSV Report",
                                data=csv_data,
                                file_name=f"model_comparison_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
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
                            mime="application/json",
                            use_container_width=True
                        )
                
                # Summary download
                st.subheader("📋 Quick Summary")
                
                summary_text = f"""
                Model Analysis Summary
                Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
                
                Models Analyzed: {', '.join(analyzer.model_results.keys())}
                
                Best Model: {analyzer.determine_best_model()[0]}
                Best F1-Score: {analyzer.determine_best_model()[1]:.4f}
                
                {'='*50}
                
                Detailed Results:
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
                    label="📝 Download Text Summary",
                    data=summary_text,
                    file_name=f"model_summary_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
            else:
                st.info("No model results to export")
        else:
            st.info("👈 Please load model results first")

if __name__ == "__main__":
    main()