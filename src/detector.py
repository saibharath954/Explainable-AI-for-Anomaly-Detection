# src/detector.py
"""
Defines the AutoencoderDetector class for anomaly detection.
"""

import numpy as np
import pandas as pd
import logging
import config
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

logger = logging.getLogger(__name__)

class AutoencoderDetector:
    """
    Wrapper for an Autoencoder anomaly detection model.
    """
    def __init__(self, input_dim, encoding_dim=14, **params):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.params = params
        self.model = self._build_model()
        logger.info(f"Autoencoder detector initialized with input_dim={input_dim}, encoding_dim={encoding_dim}")

    def _build_model(self):
        """Builds the Keras Autoencoder model."""
        input_layer = Input(shape=(self.input_dim, ))
        
        # Encoder
        encoder = Dense(self.encoding_dim, activation="tanh")(input_layer)
        encoder = Dense(int(self.encoding_dim / 2), activation="relu")(encoder)
        
        # Decoder
        decoder = Dense(self.encoding_dim, activation='tanh')(encoder)
        decoder = Dense(self.input_dim, activation='linear')(decoder)
        
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        return autoencoder

    def fit(self, X_train, X_val):
        """
        Fits the Autoencoder model.
        We train ONLY on normal data (X_train) and validate on normal data (X_val).
        """
        logger.info("Fitting Autoencoder...")
        
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        )
        
        self.model.fit(
            X_train,
            X_train,
            epochs=self.params.get('epochs', 20),
            batch_size=self.params.get('batch_size', 256),
            shuffle=True,
            validation_data=(X_val, X_val),
            callbacks=[early_stopping],
            verbose=1
        )
        logger.info("Fitted Autoencoder.")

    def get_scores(self, X):
        """
        Gets the reconstruction error (anomaly score) for new data.
        Higher error means more anomalous.
        """
        if self.model is None:
            raise RuntimeError("Detector must be fitted before getting scores.")
            
        predictions = self.model.predict(X)
        
        # Calculate Mean Squared Error (reconstruction error)
        if isinstance(X, pd.DataFrame):
            mse = np.mean(np.power(X.values - predictions, 2), axis=1)
        else:
            mse = np.mean(np.power(X - predictions, 2), axis=1)
            
        # *** IMPORTANT ***
        # Unlike Isolation Forest, for Autoencoders, a HIGH score = ANOMALY.
        # Our evaluator expects this, so no score inversion is needed.
        return mse

    def predict(self, X, threshold=0.5):
        """
        Predicts anomalies (1) or normal (0) based on a score threshold.
        """
        anomaly_scores = self.get_scores(X)
        predictions = (anomaly_scores > threshold).astype(int)
        return predictions

    def save(self, path=config.DETECTOR_PATH):
        """Saves the Keras model to disk."""
        self.model.save(path)
        logger.info(f"Detector saved to {path}")

    @classmethod
    def load(cls, path=config.DETECTOR_PATH):
        """Loads the Keras model from disk."""
        logger.info(f"Loading detector from {path}")
        loaded_model = load_model(path)
        
        # Re-create an instance to hold the model
        # input_dim will be inferred from the loaded model
        input_dim = loaded_model.layers[0].input_shape[1]
        instance = cls(input_dim=input_dim)
        instance.model = loaded_model
        return instance