from typing import Dict, Any

import tensorflow as tf

from models.kashigari_local.abc_feature_model import ABCClassificationModel
from kashgari.layers import L


class RCNN_Text_Img(ABCClassificationModel):
    def __init__(self, embedding, **params):
        super().__init__(embedding)

    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict

        activation_function list:
        {softmax, elu, selu, softplus, softsign, swish,
        relu, gelu, tanh, sigmoid, exponential,
        hard_sigmoid, linear, serialize, deserialize, get}
        """
        return {
            'layer_bilstm1': {
                'units': 256,
                'return_sequences': True
            },
            'layer_dropout': {
                'rate': 0.1,
                'name': 'layer_dropout'
            },
            'layer_output1': {
                'activation': 'softmax'
            },
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        img = tf.keras.Input(shape=(224, 224, 3), name="features")

        output_dim = self.label_processor.vocab_size
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # Define layers for BiLSTM
        layer_stack = [
            L.Bidirectional(L.LSTM(**config['layer_bilstm1'])),
            L.Dropout(**config['layer_dropout']),
            L.GlobalMaxPooling1D()
        ]
        # tensor flow in Layers {tensor:=layer(tensor)}
        tensor = embed_model.output
        for layer in layer_stack:
            tensor = layer(tensor)
        img_tensor = img
        img_stack = [
            L.Conv2D(256, (4, 4), activation='relu', padding='valid'),
            L.MaxPooling2D(),
            L.Dropout(rate=0.1),
            L.GlobalMaxPooling2D(),
        ]
        for img_layer in img_stack:
            img_tensor = img_layer(img_tensor)
        tensor = L.Attention()([tensor, tensor])
        img_tensor = L.Attention()([img_tensor, img_tensor])
        tensor = L.Concatenate(axis=-1)([img_tensor, tensor])

        output_tensor = L.Dense(output_dim, activation='softmax', name="output0")(tensor)
        self.tf_model = tf.keras.Model(inputs=[embed_model.inputs, img], outputs=output_tensor)
