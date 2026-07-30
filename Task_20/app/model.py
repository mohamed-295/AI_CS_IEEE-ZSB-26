import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import backend as K
import cv2
from PIL import Image
import io
import json
from pathlib import Path

@keras.saving.register_keras_serializable()
class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = K.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(tf.reduce_mean(loss))
        return y_pred

class OCRModel:
    def __init__(self, model_path="weights/best_ocr_model.keras"):
        full_model = tf.keras.models.load_model(
            model_path, 
            compile=False,
            custom_objects={'CTCLayer': CTCLayer}
        )
        
      
        self.model = tf.keras.Model(
            inputs=full_model.inputs[0], # <== Grab the first input tensor directly!
            outputs=full_model.get_layer('dense').output
        )
        
        self.input_shape = (80, 500, 1) 


        vocab_path = Path(__file__).resolve().parent / 'vocab.json'
        with vocab_path.open(encoding='utf-8') as f:
            self.vocab_chars = json.load(f)
        self.idx2char = {idx: char for idx, char in enumerate(self.vocab_chars)}

    def prepare_image(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = np.array(image)
        
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized_img = cv2.resize(gray_img, (self.input_shape[1], self.input_shape[0])) 
        normalized_img = resized_img / 255.0
        normalized_img = np.expand_dims(normalized_img, axis=-1)
        normalized_img = np.expand_dims(normalized_img, axis=0)
        return normalized_img

    def decode_prediction(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]

        decoded, _ = K.ctc_decode(
            pred,
            input_length=input_len,
            greedy=True
        )

        decoded = decoded[0].numpy()

        texts = []

        blank_index = len(self.vocab_chars) - 1

        for seq in decoded:
            text = ""

            for idx in seq:
                if idx == -1:
                    continue
                if idx == blank_index:
                    continue

                text += self.idx2char.get(idx, "")

            texts.append(text)

        return texts[0]
    
    def predict(self, image_bytes):
        preprocessed_img = self.prepare_image(image_bytes)
        pred = self.model.predict(preprocessed_img)
        decoded_text = self.decode_prediction(pred)
        return decoded_text