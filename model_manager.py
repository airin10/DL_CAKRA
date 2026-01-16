"""
Manager untuk semua Deep Learning models
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

class ModelManager:
    """Manage CNN, LSTM, GRU models"""
    
    def __init__(self):
        self.models = {}
        self.tokenizer = None
        self.MAX_LEN = 200
        self.IMG_SIZE = (224, 224)
        self.VOCAB_SIZE = 1000
    
    def build_cnn_model(self):
        """Build CNN model untuk image analysis"""
        model = Sequential([
            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(224, 224, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Conv Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC()]
        )
        
        return model
    
    def build_lstm_model(self):
        """Build LSTM model untuk text analysis"""
        model = Sequential([
            # Embedding layer
            layers.Embedding(self.VOCAB_SIZE, 128, 
                           input_length=self.MAX_LEN),
            
            # Bidirectional LSTM
            layers.Bidirectional(
                layers.LSTM(64, return_sequences=True)
            ),
            layers.Dropout(0.2),
            
            # Second LSTM layer
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Output
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC()]
        )
        
        return model
    
    def build_gru_model(self):
        """Build GRU model untuk text analysis"""
        model = Sequential([
            # Embedding layer
            layers.Embedding(self.VOCAB_SIZE, 128, 
                           input_length=self.MAX_LEN),
            
            # Bidirectional GRU
            layers.Bidirectional(
                layers.GRU(64, return_sequences=True)
            ),
            layers.Dropout(0.2),
            
            # Second GRU layer
            layers.Bidirectional(layers.GRU(32)),
            layers.Dropout(0.2),
            
            # Attention mechanism (simplified)
            layers.Dense(64, activation='relu'),
            layers.Attention(),
            
            # Dense layers
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Output
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC()]
        )
        
        return model
    
    def build_ensemble_model(self):
        """Build ensemble model combining all models"""
        # Ini adalah model meta yang menggabungkan predictions
        # Untuk implementasi sederhana, kita gunakan voting
        
        return {
            'cnn': self.build_cnn_model(),
            'lstm': self.build_lstm_model(),
            'gru': self.build_gru_model()
        }
    
    def preprocess_image(self, image_array):
        """Preprocess image untuk CNN"""
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        # Normalize
        image_array = image_array.astype('float32') / 255.0
        
        # Resize jika diperlukan
        if image_array.shape[1:3] != self.IMG_SIZE:
            import cv2
            resized = []
            for img in image_array:
                resized.append(cv2.resize(img, self.IMG_SIZE))
            image_array = np.array(resized)
        
        return image_array
    
    def preprocess_text(self, texts, fit_tokenizer=False):
        """Preprocess text untuk LSTM/GRU"""
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.VOCAB_SIZE, 
                                      char_level=True,
                                      lower=False)
        
        if fit_tokenizer:
            self.tokenizer.fit_on_texts(texts)
            # Save tokenizer
            with open('models/tokenizer.pkl', 'wb') as f:
                pickle.dump(self.tokenizer, f)
        
        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        padded = pad_sequences(sequences, maxlen=self.MAX_LEN, 
                              padding='post', truncating='post')
        
        return padded
    
    def train_models(self, X_train_img, X_train_txt, y_train,
                    X_val_img, X_val_txt, y_val, epochs=10):
        """Train semua model"""
        results = {}
        
        # Train CNN
        print("Training CNN model...")
        cnn_model = self.build_cnn_model()
        cnn_history = cnn_model.fit(
            self.preprocess_image(X_train_img), y_train,
            validation_data=(self.preprocess_image(X_val_img), y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        cnn_model.save('models/cnn_model.h5')
        results['cnn'] = cnn_history.history
        
        # Preprocess text
        X_train_seq = self.preprocess_text(X_train_txt, fit_tokenizer=True)
        X_val_seq = self.preprocess_text(X_val_txt)
        
        # Train LSTM
        print("Training LSTM model...")
        lstm_model = self.build_lstm_model()
        lstm_history = lstm_model.fit(
            X_train_seq, y_train,
            validation_data=(X_val_seq, y_val),
            epochs=epochs,
            batch_size=64,
            verbose=1
        )
        lstm_model.save('models/lstm_model.h5')
        results['lstm'] = lstm_history.history
        
        # Train GRU
        print("Training GRU model...")
        gru_model = self.build_gru_model()
        gru_history = gru_model.fit(
            X_train_seq, y_train,
            validation_data=(X_val_seq, y_val),
            epochs=epochs,
            batch_size=64,
            verbose=1
        )
        gru_model.save('models/gru_model.h5')
        results['gru'] = gru_history.history
        
        return results
    
    def predict_ensemble(self, image_array, text):
        """Predict menggunakan semua model"""
        predictions = {}
        
        # Load models jika belum
        if not self.models:
            self.load_all_models()
        
        # CNN prediction
        if 'cnn' in self.models:
            processed_img = self.preprocess_image(image_array)
            cnn_pred = self.models['cnn'].predict(processed_img, verbose=0)
            predictions['cnn'] = float(cnn_pred[0][0])
        
        # LSTM/GRU predictions
        if text and 'lstm' in self.models and 'gru' in self.models:
            text_seq = self.preprocess_text([text])
            
            lstm_pred = self.models['lstm'].predict(text_seq, verbose=0)
            predictions['lstm'] = float(lstm_pred[0][0])
            
            gru_pred = self.models['gru'].predict(text_seq, verbose=0)
            predictions['gru'] = float(gru_pred[0][0])
        
        # Ensemble prediction (weighted average)
        if predictions:
            weights = {'cnn': 0.4, 'lstm': 0.3, 'gru': 0.3}
            weighted_sum = 0
            total_weight = 0
            
            for model_name, pred in predictions.items():
                weight = weights.get(model_name, 0.33)
                weighted_sum += pred * weight
                total_weight += weight
            
            ensemble_pred = weighted_sum / total_weight if total_weight > 0 else 0.5
            predictions['ensemble'] = ensemble_pred
        
        return predictions
    
    def load_all_models(self):
        """Load semua model yang sudah trained"""
        try:
            self.models['cnn'] = tf.keras.models.load_model('models/cnn_model.h5')
            self.models['lstm'] = tf.keras.models.load_model('models/lstm_model.h5')
            self.models['gru'] = tf.keras.models.load_model('models/gru_model.h5')
            
            # Load tokenizer
            with open('models/tokenizer.pkl', 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False