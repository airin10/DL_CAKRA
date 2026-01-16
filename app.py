"""
QR Code Security Analyzer - Aplikasi Streamlit untuk Deteksi QR Code Berbahaya
Dibangun untuk Windows dengan Visual Studio Code
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import custom modules
from utils.qr_decoder import QRDecoder
from utils.url_analyzer import URLAnalyzer
from utils.visualizer import Visualizer
from models.cnn_model import CNNModel
from models.lstm_model import LSTMModel

# Page configuration
st.set_page_config(
    page_title="QR Code Security Analyzer",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/qr-analyzer',
        'Report a bug': "https://github.com/yourusername/qr-analyzer/issues",
        'About': "### QR Code Security Analyzer v1.0\n\nAplikasi deteksi QR code berbahaya menggunakan Deep Learning"
    }
)

# Load custom CSS
def load_css():
    """Load custom CSS dari file"""
    css_path = project_root / "assets" / "css" / "custom.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # Fallback inline CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.8rem;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 3px solid #1E3A8A;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #374151;
            margin-top: 2rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #E5E7EB;
        }
        .card {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
            border: 1px solid #E5E7EB;
        }
        .risk-high { background-color: #FEE2E2; border-left: 5px solid #DC2626; }
        .risk-medium { background-color: #FEF3C7; border-left: 5px solid #D97706; }
        .risk-low { background-color: #D1FAE5; border-left: 5px solid #059669; }
        .info-box { 
            background-color: #E0F2FE; 
            border-left: 5px solid #0EA5E9;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .stButton button {
            width: 100%;
            border-radius: 5px;
            font-weight: bold;
        }
        .stProgress > div > div > div > div {
            background-image: linear-gradient(to right, #10B981, #3B82F6, #EF4444);
        }
        </style>
        """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Load CSS
    load_css()
    
    # Initialize components
    try:
        qr_decoder = QRDecoder()
        url_analyzer = URLAnalyzer()
        visualizer = Visualizer()
        
        # Untuk demo, kita tidak perlu load model asli dulu
        # Jika ingin menggunakan model ML, uncomment berikut:
        # cnn_model = CNNModel()
        # lstm_model = LSTMModel()
        
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        st.info("Pastikan semua dependencies sudah terinstall")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/4176/4176918.png", width=100)
        st.title("🔒 QR Security Analyzer")
        
        st.markdown("---")
        
        # Navigation
        st.subheader("🔍 Mode Analisis")
        analysis_mode = st.radio(
            "",
            ["📷 Upload Gambar", "📁 Batch Processing", "🔗 Manual Input", "📊 Dashboard"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Settings")
        advanced_mode = st.toggle("Advanced Analysis", value=True)
        show_details = st.toggle("Show Detailed Results", value=True)
        
        st.markdown("---")
        
        # Quick Stats
        st.subheader("📈 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Scanned", "0")
            st.metric("Safe", "0")
        with col2:
            st.metric("Suspicious", "0")
            st.metric("Malicious", "0")
        
        st.markdown("---")
        
        # Help Section
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            1. **Upload Gambar**: Unggah gambar QR code
            2. **Batch Processing**: Analisis banyak file sekaligus
            3. **Manual Input**: Masukkan URL/teks secara manual
            4. **Dashboard**: Lihat statistik dan laporan
            
            **Tips**:
            - Pastikan gambar QR code jelas
            - Gunakan format PNG/JPG
            - File maksimal 5MB per gambar
            """)
    
    # Main Content Area
    st.markdown('<h1 class="main-header">🔒 QR Code Security Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6B7280; font-size: 1.2rem;">Deteksi dan Analisis QR Code Berbahaya menggunakan Deep Learning</p>', unsafe_allow_html=True)
    
    # Mode Selection
    if analysis_mode == "📷 Upload Gambar":
        render_upload_page(qr_decoder, url_analyzer, visualizer)
    
    elif analysis_mode == "📁 Batch Processing":
        render_batch_page(qr_decoder, url_analyzer, visualizer)
    
    elif analysis_mode == "🔗 Manual Input":
        render_manual_page(url_analyzer, visualizer)
    
    else:  # Dashboard
        render_dashboard(visualizer)

def render_upload_page(qr_decoder, url_analyzer, visualizer):
    """Render upload image page"""
    st.markdown('<h2 class="sub-header">📷 Upload Gambar QR Code</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Pilih gambar QR code",
            type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
            help="Unggah gambar yang mengandung QR code"
        )
        
        if uploaded_file is not None:
            # Display image
            from PIL import Image
            import io
            
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded QR Code", use_column_width=True)
            
            # File info
            file_size = uploaded_file.size / 1024  # Convert to KB
            st.info(f"📄 **File Info**: {uploaded_file.name} ({file_size:.1f} KB)")
            
            # Action buttons
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                analyze_btn = st.button("🔍 Analisis Sekarang", type="primary", use_container_width=True)
            with col_btn2:
                clear_btn = st.button("🗑️ Hapus", use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if analyze_btn:
                with st.spinner("🔍 Mendecode dan menganalisis QR code..."):
                    # Decode QR
                    import time
                    progress_bar = st.progress(0)
                    
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Convert PIL to bytes for decoder
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format='PNG')
                    img_bytes = img_bytes.getvalue()
                    
                    # Decode QR
                    qr_content, success = qr_decoder.decode_from_bytes(img_bytes)
                    
                    if success:
                        st.success("✅ QR Code berhasil dibaca!")
                        progress_bar.empty()
                        
                        # Analyze content
                        analysis_result = analyze_content(qr_content, url_analyzer)
                        
                        # Display results in right column
                        with col2:
                            display_analysis_results(analysis_result, qr_content, visualizer)
                    
                    else:
                        st.error("❌ Tidak dapat membaca QR code. Pastikan gambar jelas dan QR code tidak rusak.")
                        progress_bar.empty()
            
            if clear_btn:
                st.rerun()
        
        else:
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show examples
            with st.expander("📋 Contoh Format yang Didukung", expanded=True):
                st.markdown("""
                | Format | Ukuran Maks | Catatan |
                |--------|------------|---------|
                | PNG    | 5 MB       | Disarankan |
                | JPG/JPEG | 5 MB     | Pastikan tidak blur |
                | BMP    | 5 MB       | Lossless format |
                | GIF    | 5 MB       | Hanya frame pertama |
                
                **Tips**:
                - Gunakan gambar dengan resolusi minimal 300x300px
                - Pastikan QR code tidak terpotong
                - Hindari cahaya silau/refleksi
                """)
    
    with col2:
        if 'analysis_result' not in st.session_state:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.info("👈 Upload gambar QR code untuk memulai analisis")
            st.markdown("</div>", unsafe_allow_html=True)

def render_batch_page(qr_decoder, url_analyzer, visualizer):
    """Render batch processing page"""
    st.markdown('<h2 class="sub-header">📁 Batch Processing</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Pilih multiple file atau folder (zip)",
            type=['png', 'jpg', 'jpeg', 'bmp', 'zip'],
            accept_multiple_files=True,
            help="Pilih beberapa file sekaligus atau upload file ZIP"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file berhasil diupload")
            
            # File list
            with st.expander("📁 Daftar File", expanded=True):
                for file in uploaded_files:
                    file_size = file.size / 1024
                    st.write(f"📄 {file.name} ({file_size:.1f} KB)")
            
            # Batch settings
            st.subheader("⚙️ Batch Settings")
            col_set1, col_set2 = st.columns(2)
            with col_set1:
                parallel_processing = st.checkbox("Parallel Processing", value=True)
                save_results = st.checkbox("Save Results", value=True)
            with col_set2:
                export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"])
                confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
            
            # Process button
            if st.button("🚀 Proses Batch", type="primary", use_container_width=True):
                process_batch_files(uploaded_files, qr_decoder, url_analyzer, export_format)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Batch Statistics")
        
        # Statistics placeholders
        metrics_data = {
            "Total Files": 0,
            "Processed": 0,
            "Successful": 0,
            "Failed": 0,
            "Safe": 0,
            "Suspicious": 0,
            "Malicious": 0
        }
        
        for key, value in metrics_data.items():
            st.metric(key, value)
        
        st.markdown("---")
        st.caption("Upload file untuk melihat statistik")
        st.markdown('</div>', unsafe_allow_html=True)

def render_manual_page(url_analyzer, visualizer):
    """Render manual input page"""
    st.markdown('<h2 class="sub-header">🔗 Manual Input Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        input_method = st.radio(
            "Pilih input method:",
            ["URL", "Plain Text", "QR Code Text"],
            horizontal=True
        )
        
        if input_method == "URL":
            url_input = st.text_input(
                "Masukkan URL:",
                placeholder="https://example.com",
                help="Masukkan URL lengkap dengan http:// atau https://"
            )
            
            if url_input:
                # Generate QR preview
                try:
                    import qrcode
                    qr = qrcode.QRCode(
                        version=1,
                        error_correction=qrcode.constants.ERROR_CORRECT_L,
                        box_size=10,
                        border=4,
                    )
                    qr.add_data(url_input)
                    qr.make(fit=True)
                    
                    img = qr.make_image(fill_color="black", back_color="white")
                    st.image(img, caption="QR Code Preview", width=200)
                    
                    # Analyze button
                    if st.button("🔍 Analisis URL", type="primary"):
                        analyze_manual_input(url_input, url_analyzer, visualizer, col2)
                
                except ImportError:
                    st.warning("Install qrcode library: pip install qrcode[pil]")
        
        elif input_method == "Plain Text":
            text_input = st.text_area(
                "Masukkan teks:",
                placeholder="Masukkan teks yang ingin dianalisis...",
                height=100
            )
            
            if text_input and st.button("🔍 Analisis Teks", type="primary"):
                analyze_manual_input(text_input, url_analyzer, visualizer, col2)
        
        else:  # QR Code Text
            qr_text = st.text_area(
                "Tempel teks dari QR code:",
                placeholder="Teks hasil scan QR code...",
                height=150
            )
            
            if qr_text and st.button("🔍 Analisis QR Text", type="primary"):
                analyze_manual_input(qr_text, url_analyzer, visualizer, col2)
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_dashboard(visualizer):
    """Render dashboard page"""
    st.markdown('<h2 class="sub-header">📊 Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Scans", "1,247", "+12%")
    with col2:
        st.metric("Detection Rate", "94.3%", "+2.1%")
    with col3:
        st.metric("Avg Risk Score", "32.5", "-5.2")
    with col4:
        st.metric("Threats Blocked", "187", "+8")
    
    st.markdown("---")
    
    # Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📈 Risk Distribution")
        # Placeholder for chart
        st.info("Chart akan muncul setelah ada data")
    
    with col_chart2:
        st.subheader("📊 Threat Categories")
        # Placeholder for chart
        st.info("Chart akan muncul setelah ada data")
    
    # Recent scans table
    st.subheader("🕐 Recent Scans")
    # Placeholder table
    import pandas as pd
    dummy_data = pd.DataFrame({
        'Time': ['10:30', '10:25', '10:20', '10:15', '10:10'],
        'URL': ['example.com', 'test.com', 'secure.com', 'bank.com', 'pay.com'],
        'Risk': ['Low', 'Medium', 'High', 'Low', 'Medium'],
        'Action': ['Allowed', 'Blocked', 'Blocked', 'Allowed', 'Warned']
    })
    st.dataframe(dummy_data, use_container_width=True)

def analyze_content(content, url_analyzer):
    """Analyze QR code content"""
    import time
    
    # Simulate analysis process
    analysis_result = {
        'content': content,
        'is_url': content.startswith(('http://', 'https://')) if content else False,
        'risk_score': 0,
        'risk_level': 'unknown',
        'confidence': 0.0,
        'features': {},
        'reasons': [],
        'recommendations': []
    }
    
    # Analyze if it's a URL
    if analysis_result['is_url']:
        url_features = url_analyzer.extract_features(content)
        risk_assessment = url_analyzer.assess_risk(url_features)
        
        analysis_result.update({
            'risk_score': risk_assessment['risk_score'],
            'risk_level': risk_assessment['risk_level'],
            'confidence': risk_assessment['confidence'],
            'features': url_features,
            'reasons': risk_assessment['reasons'],
            'recommendations': risk_assessment['recommendations']
        })
    else:
        # Analyze plain text
        text_risk = len(content) / 1000  # Simple risk based on length
        analysis_result['risk_score'] = min(100, text_risk * 100)
        
        if text_risk > 0.7:
            analysis_result['risk_level'] = 'high'
            analysis_result['reasons'] = ["Teks sangat panjang (mungkin encoded data)"]
        elif text_risk > 0.3:
            analysis_result['risk_level'] = 'medium'
            analysis_result['reasons'] = ["Teks panjang, perlu pemeriksaan manual"]
        else:
            analysis_result['risk_level'] = 'low'
            analysis_result['reasons'] = ["Teks pendek, kemungkinan aman"]
        
        analysis_result['recommendations'] = ["Periksa isi teks sebelum digunakan"]
        analysis_result['confidence'] = 0.5
    
    return analysis_result

def display_analysis_results(analysis_result, qr_content, visualizer):
    """Display analysis results"""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Risk indicator
    risk_class = f"risk-{analysis_result['risk_level']}"
    risk_labels = {
        'high': '🚨 HIGH RISK',
        'medium': '⚠️ MEDIUM RISK', 
        'low': '✅ LOW RISK',
        'unknown': '❓ UNKNOWN'
    }
    
    st.markdown(f"""
    <div class="{risk_class}" style="padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
        <h3 style="margin: 0 0 0.5rem 0;">{risk_labels.get(analysis_result['risk_level'], 'UNKNOWN')}</h3>
        <div style="display: flex; justify-content: space-between;">
            <div>Risk Score: <strong>{analysis_result['risk_score']:.1f}/100</strong></div>
            <div>Confidence: <strong>{analysis_result['confidence']:.1f}%</strong></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk progress bar
    risk_percentage = analysis_result['risk_score'] / 100
    st.progress(risk_percentage)
    st.caption(f"Risk Level: {risk_percentage:.0%}")
    
    # Content preview
    with st.expander("📝 Content Preview", expanded=True):
        st.code(qr_content if qr_content else "No content", language="text")
    
    # Analysis details
    if analysis_result['reasons']:
        st.subheader("🔍 Analysis Details")
        for reason in analysis_result['reasons']:
            st.write(f"• {reason}")
    
    if analysis_result['recommendations']:
        st.subheader("📋 Recommendations")
        for rec in analysis_result['recommendations']:
            st.write(f"• {rec}")
    
    # URL features if applicable
    if analysis_result['is_url'] and analysis_result['features']:
        with st.expander("🔗 URL Features", expanded=False):
            features_df = visualizer.create_features_table(analysis_result['features'])
            st.dataframe(features_df, use_container_width=True)
    
    # Action buttons
    st.markdown("---")
    col_act1, col_act2, col_act3 = st.columns(3)
    
    with col_act1:
        st.button("📋 Copy Result", use_container_width=True)
    with col_act2:
        st.button("📤 Export", use_container_width=True)
    with col_act3:
        st.button("🔄 Analyze Again", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Store in session state
    st.session_state['analysis_result'] = analysis_result

def process_batch_files(files, qr_decoder, url_analyzer, export_format):
    """Process batch of files"""
    import pandas as pd
    from datetime import datetime
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    for i, file in enumerate(files):
        status_text.text(f"Processing {file.name} ({i+1}/{len(files)})")
        
        try:
            # Read file
            if file.type.startswith('image/'):
                # Decode QR
                qr_content, success = qr_decoder.decode_from_bytes(file.getvalue())
                
                if success:
                    # Analyze
                    analysis = analyze_content(qr_content, url_analyzer)
                    
                    results.append({
                        'filename': file.name,
                        'content_preview': qr_content[:50] + '...' if len(qr_content) > 50 else qr_content,
                        'status': 'Success',
                        'risk_score': analysis['risk_score'],
                        'risk_level': analysis['risk_level'],
                        'confidence': analysis['confidence']
                    })
                else:
                    results.append({
                        'filename': file.name,
                        'content_preview': 'N/A',
                        'status': 'Failed to decode',
                        'risk_score': 0,
                        'risk_level': 'unknown',
                        'confidence': 0
                    })
        
        except Exception as e:
            results.append({
                'filename': file.name,
                'content_preview': f'Error: {str(e)[:50]}',
                'status': 'Error',
                'risk_score': 0,
                'risk_level': 'unknown',
                'confidence': 0
            })
        
        progress_bar.progress((i + 1) / len(files))
    
    status_text.text("Processing complete!")
    
    # Display results
    if results:
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Export options
        if export_format == "CSV":
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"qr_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        elif export_format == "JSON":
            json_str = results_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"qr_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def analyze_manual_input(content, url_analyzer, visualizer, results_column):
    """Analyze manual input"""
    with results_column:
        with st.spinner("Analyzing..."):
            analysis_result = analyze_content(content, url_analyzer)
            display_analysis_results(analysis_result, content, visualizer)

if __name__ == "__main__":
    main()