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
    def __init__(self, input_dim, **params):
        self.input_dim = input_dim
        self.params = params
        
        # Get layer structure from params.
        # Default to the *original* structure [14, 7] if not provided
        # (e.g., this handles the case when loading a model)
        self.layers = self.params.get('layers', [14, 7])
        
        if 'layers' not in self.params:
            logger.warning(
                f"Autoencoder 'layers' not specified in params. "
                f"Defaulting to {self.layers}. (This is normal during model load)."
            )
            
        self.model = self._build_model()
        logger.info(f"Autoencoder detector initialized with input_dim={input_dim}, layers={self.layers}")

    def _build_model(self):
        """
        Builds the Keras Autoencoder model dynamically based on self.layers.
        Uses 'relu' for hidden layers for better deep network training.
        """
        input_layer = Input(shape=(self.input_dim, ))
        
        x = input_layer
        
        # --- Encoder ---
        # Build encoder layers (e.g., [20, 10, 5])
        for layer_dim in self.layers:
            x = Dense(layer_dim, activation="relu")(x)
        
        # 'x' is now the bottleneck layer
        
        # --- Decoder ---
        # Build decoder layers in reverse (e.g., [10, 20])
        # We reverse all layers *except* the last one (the bottleneck)
        for layer_dim in reversed(self.layers[:-1]):
            x = Dense(layer_dim, activation="relu")(x)
        
        # --- Output Layer ---
        # Reconstruct back to the original input_dim
        output_layer = Dense(self.input_dim, activation='linear')(x) 
        
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
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
        # We need to pass custom_objects=None if we don't have custom layers/losses
        # but the default load_model should be fine here.
        loaded_model = load_model(path) 
        
        # Re-create an instance to hold the model
        # input_dim will be inferred from the loaded model
        input_dim = loaded_model.layers[0].input_shape[1]
        
        # This will call __init__ with an empty params dict,
        # which will use the default [14, 7] layers to build a
        # temporary model. This is fine.
        instance = cls(input_dim=input_dim) 
        
        # We immediately replace the temporary model with the loaded one.
        instance.model = loaded_model
        return instance