"""
QR Code Decoder Module
"""

import cv2
import numpy as np
from PIL import Image
import io
from pyzbar import pyzbar

class QRDecoder:
    """Class untuk decode QR code dari berbagai format"""
    
    def __init__(self):
        self.decode_history = []
    
    def decode_from_bytes(self, image_bytes):
        """Decode QR code dari bytes"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return None, False
            
            # Decode QR code
            decoded = pyzbar.decode(img)
            
            if decoded:
                content = decoded[0].data.decode('utf-8')
                self.decode_history.append({
                    'content': content,
                    'success': True,
                    'timestamp': np.datetime64('now')
                })
                return content, True
            else:
                # Try with preprocessing
                processed = self._preprocess_image(img)
                decoded = pyzbar.decode(processed)
                
                if decoded:
                    content = decoded[0].data.decode('utf-8')
                    self.decode_history.append({
                        'content': content,
                        'success': True,
                        'timestamp': np.datetime64('now')
                    })
                    return content, True
                else:
                    self.decode_history.append({
                        'content': None,
                        'success': False,
                        'timestamp': np.datetime64('now')
                    })
                    return None, False
        
        except Exception as e:
            print(f"Decode error: {str(e)}")
            return None, False
    
    def decode_from_file(self, file_path):
        """Decode QR code dari file"""
        try:
            img = cv2.imread(file_path)
            if img is None:
                return None, False
            
            decoded = pyzbar.decode(img)
            
            if decoded:
                content = decoded[0].data.decode('utf-8')
                return content, True
            return None, False
        
        except Exception as e:
            print(f"File decode error: {str(e)}")
            return None, False
    
    def decode_from_pil(self, pil_image):
        """Decode QR code dari PIL Image"""
        try:
            # Convert PIL to OpenCV
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            decoded = pyzbar.decode(opencv_image)
            
            if decoded:
                content = decoded[0].data.decode('utf-8')
                return content, True
            return None, False
        
        except Exception as e:
            print(f"PIL decode error: {str(e)}")
            return None, False
    
    def _preprocess_image(self, image):
        """Preprocess image untuk decoding yang lebih baik"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
            
            return blurred
        
        except:
            return image
    
    def get_statistics(self):
        """Get decoding statistics"""
        if not self.decode_history:
            return {'total': 0, 'success_rate': 0}
        
        total = len(self.decode_history)
        successful = sum(1 for entry in self.decode_history if entry['success'])
        success_rate = (successful / total) * 100 if total > 0 else 0
        
        return {
            'total': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': success_rate
        }