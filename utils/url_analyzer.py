"""
URL Analysis Module
"""

import re
import tldextract
from urllib.parse import urlparse, parse_qs
import ipaddress

class URLAnalyzer:
    """Class untuk analisis URL dari QR code"""
    
    def __init__(self):
        self.suspicious_patterns = [
            # Suspicious keywords
            ('login', 15),
            ('secure', 10),
            ('account', 15),
            ('verify', 20),
            ('bank', 25),
            ('pay', 20),
            ('password', 30),
            ('update', 15),
            ('confirm', 15),
            ('signin', 20),
            
            # URL shorteners
            ('bit.ly', 20),
            ('tinyurl', 20),
            ('goo.gl', 20),
            ('ow.ly', 20),
            ('is.gd', 20),
            
            # Special characters patterns
            ('@', 25),  # Email in URL
            ('//', 10),  # Double slash
            ('..', 30),  # Directory traversal
            ('%', 15),   # URL encoded
        ]
        
        self.safe_domains = [
            'google.com', 'youtube.com', 'facebook.com',
            'wikipedia.org', 'amazon.com', 'twitter.com',
            'instagram.com', 'linkedin.com', 'microsoft.com',
            'apple.com', 'github.com'
        ]
    
    def extract_features(self, url):
        """Extract features dari URL"""
        features = {}
        
        if not url:
            return features
        
        # Basic features
        features['length'] = len(url)
        features['has_https'] = url.startswith('https://')
        features['has_http'] = url.startswith('http://')
        features['has_www'] = 'www.' in url.lower()
        
        # Extract domain info
        try:
            extracted = tldextract.extract(url)
            features['domain'] = f"{extracted.domain}.{extracted.suffix}"
            features['subdomain'] = extracted.subdomain
            features['subdomain_count'] = len(extracted.subdomain.split('.')) if extracted.subdomain else 0
            
            # Check if domain is in safe list
            features['is_known_safe'] = features['domain'] in self.safe_domains
        except:
            features['domain'] = "unknown"
            features['is_known_safe'] = False
        
        # Check for IP address
        features['has_ip'] = self._contains_ip(url)
        
        # Check for suspicious patterns
        suspicious_score = 0
        suspicious_found = []
        
        for pattern, score in self.suspicious_patterns:
            if pattern in url.lower():
                suspicious_score += score
                suspicious_found.append(pattern)
        
        features['suspicious_score'] = suspicious_score
        features['suspicious_patterns'] = suspicious_found
        
        # Extract query parameters
        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            features['query_param_count'] = len(query_params)
            features['has_query'] = bool(parsed.query)
        except:
            features['query_param_count'] = 0
            features['has_query'] = False
        
        # Check for special characters
        special_chars = len(re.findall(r'[^\w\s./:-]', url))
        features['special_char_count'] = special_chars
        
        return features
    
    def assess_risk(self, features):
        """Assess risk berdasarkan features"""
        risk_score = 0
        reasons = []
        
        # Length check
        if features.get('length', 0) > 100:
            risk_score += 20
            reasons.append("URL terlalu panjang")
        
        # HTTPS check
        if not features.get('has_https', False):
            risk_score += 25
            reasons.append("Menggunakan HTTP (tidak aman)")
        else:
            risk_score -= 10  # HTTPS mengurangi risiko
            reasons.append("Menggunakan HTTPS (lebih aman)")
        
        # Known safe domain
        if features.get('is_known_safe', False):
            risk_score -= 30
            reasons.append("Domain dikenal aman")
        
        # Suspicious patterns
        if features.get('suspicious_score', 0) > 0:
            risk_score += features['suspicious_score']
            patterns = features.get('suspicious_patterns', [])
            if patterns:
                reasons.append(f"Pola mencurigakan: {', '.join(patterns[:3])}")
        
        # IP address in URL
        if features.get('has_ip', False):
            risk_score += 40
            reasons.append("Mengandung alamat IP langsung")
        
        # Many query parameters
        if features.get('query_param_count', 0) > 5:
            risk_score += 15
            reasons.append("Banyak parameter query")
        
        # Special characters
        if features.get('special_char_count', 0) > 10:
            risk_score += 20
            reasons.append("Banyak karakter khusus")
        
        # Normalize score
        risk_score = max(0, min(100, risk_score))
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = 'high'
        elif risk_score >= 40:
            risk_level = 'medium'
        elif risk_score >= 20:
            risk_level = 'low'
        else:
            risk_level = 'very_low'
        
        # Calculate confidence
        confidence = 100 - risk_score
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risk_level, features)
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'confidence': confidence,
            'reasons': reasons,
            'recommendations': recommendations,
            'features': features
        }
    
    def _contains_ip(self, url):
        """Check if URL contains IP address"""
        # Find all potential IP addresses
        ip_patterns = [
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  # IPv4
            r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'  # IPv6 (simplified)
        ]
        
        for pattern in ip_patterns:
            if re.search(pattern, url):
                return True
        return False
    
    def _generate_recommendations(self, risk_level, features):
        """Generate recommendations berdasarkan risk level"""
        recommendations = []
        
        if risk_level == 'high':
            recommendations = [
                "🚨 JANGAN akses URL ini!",
                "Hindari memasukkan informasi pribadi",
                "Laporkan ke administrator jika ditemukan di lingkungan kerja",
                "Scan dengan antivirus jika sudah terlanjur mengklik"
            ]
        elif risk_level == 'medium':
            recommendations = [
                "⚠️ Hati-hati saat mengakses",
                "Verifikasi sumber QR code",
                "Jangan masukkan data sensitif",
                "Gunakan browser dalam mode aman"
            ]
        elif risk_level == 'low':
            recommendations = [
                "✅ Relatif aman, tapi tetap waspada",
                "Periksa URL sebelum login",
                "Pastikan koneksi aman (HTTPS)",
                "Update browser dan antivirus"
            ]
        else:  # very_low
            recommendations = [
                "✅ URL tampak aman",
                "Tetap gunakan praktik keamanan standar",
                "Perbarui bookmark jika sering diakses"
            ]
        
        # Additional specific recommendations
        if not features.get('has_https', False):
            recommendations.append("🔒 Gunakan versi HTTPS jika tersedia")
        
        if features.get('has_ip', False):
            recommendations.append("🌐 Hindari URL dengan alamat IP langsung")
        
        if features.get('query_param_count', 0) > 3:
            recommendations.append("🔍 Periksa parameter URL yang tidak biasa")
        
        return recommendations