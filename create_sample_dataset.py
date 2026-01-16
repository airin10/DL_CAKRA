"""
Script untuk membuat sample dataset dengan struktur benign/malicious
"""

import os
import zipfile
from PIL import Image
import numpy as np
import qrcode

def create_sample_qr_codes():
    """Create sample QR codes untuk testing"""
    
    # Buat folder structure
    os.makedirs('sample_dataset/benign', exist_ok=True)
    os.makedirs('sample_dataset/malicious', exist_ok=True)
    
    # Safe URLs (benign)
    benign_urls = [
        "https://www.google.com",
        "https://www.github.com",
        "https://www.wikipedia.org",
        "WIFI:S:MySafeNetwork;T:WPA2;P:StrongPassword123;;",
        "mailto:support@example.com",
        "tel:+1234567890",
        "https://www.linkedin.com",
        "https://www.youtube.com",
        "BEGIN:VCARD\nVERSION:3.0\nFN:John Doe\nORG:Company\nTEL:+1234567890\nEMAIL:john@example.com\nEND:VCARD",
        "https://www.amazon.com"
    ]
    
    # Malicious URLs
    malicious_urls = [
        "http://free-gift-cards.com/login.php",
        "javascript:alert('XSS Attack')",
        "data:text/html;base64,PHNjcmlwdD5hbGVydCgnTWFsaWNpb3VzJyk8L3NjcmlwdD4=",
        "http://192.168.1.1:8080/admin",
        "https://bit.ly/suspicious-link-xyz123",
        "http://phishing-bank-site.com/verify",
        "Download free virus: http://malware-exe.com/file.exe",
        "You won $1,000,000! Claim: http://fake-lottery.com",
        "Your account is locked! Reset: http://fake-security-site.com",
        "Free Bitcoin generator: http://scam-crypto-site.com"
    ]
    
    print("Creating benign QR codes...")
    for i, content in enumerate(benign_urls):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(content)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(f'sample_dataset/benign/qr_{i+1}.png')
    
    print("Creating malicious QR codes...")
    for i, content in enumerate(malicious_urls):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(content)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(f'sample_dataset/malicious/mal_qr_{i+1}.png')
    
    print("Creating ZIP file...")
    with zipfile.ZipFile('sample_dataset.zip', 'w') as zipf:
        for folder in ['benign', 'malicious']:
            for file in os.listdir(f'sample_dataset/{folder}'):
                zipf.write(f'sample_dataset/{folder}/{file}', 
                          f'{folder}/{file}')
    
    # Cleanup
    shutil.rmtree('sample_dataset')
    
    print("✅ Sample dataset created: sample_dataset.zip")
    print("You can upload this to the Streamlit app for testing!")

if __name__ == "__main__":
    import shutil
    create_sample_qr_codes()