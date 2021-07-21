
'''
author: FoVNull
@hikki.top
based on huggingface@https://huggingface.co/transformers/model_doc/mpnet.html
'''

from typing import Dict, List, Any, Optional
import os
import json
import codecs

from transformers import TFBertModel
import tensorflow as tf
import tensorflow.keras.layers as L

from kashgari.embeddings.abc_embedding import ABCEmbedding
from kashgari.logger import logger


class HFBertEmbedding(ABCEmbedding):
    def to_dict(self) -> Dict[str, Any]:
        info_dic = super(HFBertEmbedding, self).to_dict()
        info_dic['config']['path'] = self.path
        return info_dic

    def __init__(self,
                 path: str,
                 **kwargs: Any):
        self.path = path
        self.config_path = path + '/config.json'
        self.vocab_path = path + '/vocab.txt'
        self.vocab_list: List[str] = []
        kwargs['segment'] = True
        super(HFBertEmbedding, self).__init__(**kwargs)

    def load_embed_vocab(self) -> Optional[Dict[str, int]]:
        token2idx: Dict[str, int] = {}
        with codecs.open(self.vocab_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.vocab_list.append(token)
                token2idx[token] = len(token2idx)
        top_words = [k for k, v in list(token2idx.items())[:50]]
        logger.debug('------------------------------------------------')
        logger.debug("Loaded vocab")
        logger.debug(f'config_path       : {self.config_path}')
        logger.debug(f'vocab_path      : {self.vocab_path}')
        logger.debug(f'Top 50 words    : {top_words}')
        logger.debug('------------------------------------------------')

        return token2idx

    def build_embedding_model(self,
                              *,
                              vocab_size: int = None,
                              force: bool = False,
                              **kwargs: Dict) -> None:
        if self.embed_model is None:
            model = self.create_model()
            for layer in model.layers:
                layer.trainable = False
            self.embed_model = model
            self.embedding_size = model.output.shape[-1]

    def create_model(self):
        inputs = tf.keras.Input(shape=(None, ), name='data', dtype='int32')
        targets = tf.keras.Input(shape=(None,), name='output', dtype='int32')

        hf_model = TFBertModel.from_pretrained(self.path)
        encodings = hf_model(inputs).hidden_states[2]

        model = tf.keras.Model(inputs=[inputs, targets], outputs=encodings)

        return model
