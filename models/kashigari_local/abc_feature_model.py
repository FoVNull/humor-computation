# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: abs_model.py
# time: 4:05 下午

from abc import ABC
import numpy as np
from typing import List, Dict, Any, Union

from sklearn import metrics as sklearn_metrics
import pickle
from tensorflow import keras
import tensorflow as tf

import kashgari
from kashgari.embeddings import ABCEmbedding, BareEmbedding
from models.kashigari_local.generators import CorpusGenerator, CorpusFeaturesGenerator, BatchDataSetFeatures
from kashgari.layers import L
from kashgari.logger import logger
from kashgari.metrics.multi_label_classification import multi_label_classification_report
from kashgari.processors import ABCProcessor
from kashgari.processors import ClassificationProcessor
from kashgari.processors import SequenceProcessor
from kashgari.tasks.abs_task_model import ABCTaskModel
from kashgari.types import TextSamplesVar, ClassificationLabelVar, MultiLabelClassificationLabelVar


class ABCClassificationModel(ABCTaskModel, ABC):
    """
    Abstract Classification Model
    """

    __task__ = 'classification'

    def to_dict(self) -> Dict:
        info = super(ABCClassificationModel, self).to_dict()
        info['config']['multi_label'] = self.multi_label
        return info

    def __init__(self,
                 embedding: ABCEmbedding = None,
                 *,
                 sequence_length: int = None,
                 hyper_parameters: Dict[str, Dict[str, Any]] = None,
                 multi_label: bool = False,
                 task_num: int = 1,
                 text_processor: ABCProcessor = None,
                 label_processor: ABCProcessor = None):
        """

        Args:
            embedding: embedding object
            sequence_length: target sequence length
            hyper_parameters: hyper_parameters to overwrite
            multi_label: is multi-label classification
            text_processor: text processor
            label_processor: label processor
        """
        super(ABCClassificationModel, self).__init__()
        if embedding is None:
            embedding = BareEmbedding()  # type: ignore

        if hyper_parameters is None:
            hyper_parameters = self.default_hyper_parameters()

        if text_processor is None:
            text_processor = SequenceProcessor()

        if label_processor is None:
            label_processor = ClassificationProcessor(multi_label=multi_label)
            # 判断是否为多任务
            if task_num > 1:
                '''
                这里设定multi_label为True是为了将input转为one-hot形式
                从而使用binary_crossentropy或categorical_crossentropy(效果相较sparse_categorical_crossentropy更佳)
                实际上并不是为了多标签，而是多任务多输出，每个任务单标签
                所以evaluate时需要有所改动
                '''
                label_processor = [ClassificationProcessor(multi_label=True) for _ in range(task_num)]

        self.tf_model: keras.Model = None
        self.embedding = embedding
        self.hyper_parameters = hyper_parameters
        self.sequence_length = sequence_length
        self.multi_label = multi_label
        self.task_num = task_num

        self.text_processor = text_processor
        self.label_processor = label_processor

    def _activation_layer(self) -> L.Layer:
        if self.multi_label:
            return L.Activation('sigmoid')
        else:
            return L.Activation('softmax')

    def build_model(self,
                    x_train: TextSamplesVar,
                    y_train: Union[ClassificationLabelVar, MultiLabelClassificationLabelVar]) -> None:
        """
        Build Model with x_data and y_data

        This function will setup a :class:`CorpusGenerator`,
         then call py:meth:`ABCClassificationModel.build_model_gen` for preparing processor and model

        Args:
            x_train:
            y_train:

        Returns:

        """
        self.build_model_generator([CorpusGenerator(x_train, [y[i] for y in y_train]) for i in range(self.task_num)])

    def build_model_generator(self,
                              generators: List[CorpusGenerator]) -> None:
        '''
        这里说明一下 text_processor只处理x_data不处理标签
        多个标签分类任务情况下，由于文本表示为多任务共享，build_vocab_generator只对一组generators进行执行
        多任务不同的文本表示功能，暂未实现
        '''
        if not self.text_processor.vocab2idx:
            self.text_processor.build_vocab_generator(generators[0])

        if self.task_num == 1:
            self.label_processor.build_vocab_generator(generators[0])
        else:
            for i, lp in enumerate(self.label_processor):
                lp.build_vocab_generator(generators[i])

        self.embedding.setup_text_processor(self.text_processor)

        if self.sequence_length is None:
            # 同上，多个任务共享一组x_data的generators即可
            self.sequence_length = self.embedding.get_seq_length_from_corpus(generators[0])

        if self.tf_model is None:
            self.build_model_arc()
            self.compile_model()

    def build_model_arc(self) -> None:
        raise NotImplementedError

    def compile_model(self,
                      loss: Any = None,
                      optimizer: Any = None,
                      metrics: Any = None,
                      **kwargs: Any) -> None:
        """
        Configures the model for training.
        call :meth:`tf.keras.Model.predict` to compile model with custom loss, optimizer and metrics

        Examples:

            >>> model = BiLSTM_Model()
            # Build model with corpus
            >>> model.build_model(train_x, train_y)
            # Compile model with custom loss, optimizer and metrics
            >>> model.compile(loss='categorical_crossentropy', optimizer='rsm', metrics = ['accuracy'])

        Args:
            loss: name of objective function, objective function or ``tf.keras.losses.Loss`` instance.
            optimizer: name of optimizer or optimizer instance.
            metrics (object): List of metrics to be evaluated by the model during training and testing.
            **kwargs: additional params passed to :meth:`tf.keras.Model.predict``.
        """
        if loss is None:
            '''
            loss函数选择
            '''
            if self.multi_label:
                loss = 'binary_crossentropy'
            else:
                loss = 'sparse_categorical_crossentropy'

            loss_weights = None
            if self.task_num > 1:
                loss = {}
                loss_weights = {}
                for i in range(self.task_num):
                    loss['output' + str(i)] = 'binary_crossentropy'
                    if i == 0:
                        loss_weights['output' + str(i)] = 1.
                    else:
                        loss_weights['output' + str(i)] = 0.8
        if optimizer is None:
            optimizer = 'adam'
        if metrics is None:
            metrics = ['accuracy']

        self.tf_model.compile(loss=loss,
                              loss_weights=loss_weights,
                              optimizer=optimizer,
                              metrics=metrics,
                              **kwargs)

    def fit(self,
            x_train,
            y_train: Union[ClassificationLabelVar, MultiLabelClassificationLabelVar],
            x_validate=None,
            y_validate: Union[ClassificationLabelVar, MultiLabelClassificationLabelVar] = None,
            *,
            batch_size: int = 64,
            epochs: int = 5,
            callbacks: List['keras.callbacks.Callback'] = None,
            fit_kwargs: Dict = None) -> 'keras.callbacks.History':
        """
        Trains the model for a given number of epochs with given data set list.

        Args:
            x_train: Array of train feature data (if the model has a single input),
                or tuple of train feature data array (if the model has multiple inputs)
            y_train: Array of train label data
            x_validate: Array of validation feature data (if the model has a single input),
                or tuple of validation feature data array (if the model has multiple inputs)
            y_validate: Array of validation label data
            batch_size: Number of samples per gradient update, default to 64.
            epochs: Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y` data provided.
            callbacks: List of `tf.keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See :class:`tf.keras.callbacks`.
            fit_kwargs: fit_kwargs: additional arguments passed to :meth:`tf.keras.Model.fit`

        Returns:
            A :class:`tf.keras.callback.History`  object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        """
        train_gen = CorpusFeaturesGenerator(x_train, y_train)
        train_non_feature_gen = [CorpusGenerator(x_train[0], [y[i] for y in y_train]) for i in range(self.task_num)]

        if x_validate is not None:
            valid_gen = CorpusFeaturesGenerator(x_validate, y_validate)
            valid_non_feature_gen = [CorpusGenerator(x_validate[0], [y[i] for y in y_validate]) for i in
                                     range(self.task_num)]
        else:
            valid_gen = None
            valid_non_feature_gen = None
        return self.fit_generator(train_sample_gen=train_gen,
                                  train_non_feature_gen=train_non_feature_gen,
                                  valid_sample_gen=valid_gen,
                                  valid_non_feature_gen=valid_non_feature_gen,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  fit_kwargs=fit_kwargs)

    def fit_generator(self,
                      train_sample_gen: CorpusFeaturesGenerator,
                      train_non_feature_gen: List[CorpusGenerator],
                      valid_sample_gen: CorpusFeaturesGenerator = None,
                      valid_non_feature_gen: List[CorpusGenerator] = None,
                      *,
                      batch_size: int = 64,
                      epochs: int = 5,
                      callbacks: List['keras.callbacks.Callback'] = None,
                      fit_kwargs: Dict = None) -> 'keras.callbacks.History':
        """
        Trains the model for a given number of epochs with given data generator.

        Data generator must be the subclass of `CorpusGenerator`

        Args:
            train_sample_gen: train data generator.
            valid_sample_gen: valid data generator.
            batch_size: Number of samples per gradient update, default to 64.
            epochs: Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y` data provided.
            callbacks: List of `tf.keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See `tf.keras.callbacks`.
            fit_kwargs: fit_kwargs: additional arguments passed to :meth:`tf.keras.Model.fit`

        Returns:
            A :py:class:`tf.keras.callback.History`  object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        """
        '''
        build_model_generator的输入shape=(task_num, None)
        每次输入一个子任务的 训练、验证（验证集可为空）
        代码比之前啰嗦了一点
        '''
        if valid_non_feature_gen is None:
            self.build_model_generator([[train_non_feature_gen[i]] for i in range(self.task_num)])
        else:
            self.build_model_generator(
                [[train_non_feature_gen[i], valid_non_feature_gen[i]] for i in range(self.task_num)])

        model_summary = []
        self.tf_model.summary(print_fn=lambda x: model_summary.append(x))
        logger.debug('\n'.join(model_summary))

        train_set = BatchDataSetFeatures(train_sample_gen,
                                         text_processor=self.text_processor,
                                         label_processor=self.label_processor,
                                         segment=self.embedding.segment,
                                         seq_length=self.sequence_length,
                                         batch_size=batch_size)
        if fit_kwargs is None:
            fit_kwargs = {}

        if valid_sample_gen:
            valid_gen = BatchDataSetFeatures(valid_sample_gen,
                                             text_processor=self.text_processor,
                                             label_processor=self.label_processor,
                                             segment=self.embedding.segment,
                                             seq_length=self.sequence_length,
                                             batch_size=batch_size)
            fit_kwargs['validation_data'] = valid_gen.take()
            fit_kwargs['validation_steps'] = len(valid_gen)
        return self.tf_model.fit(train_set.take(),
                                 steps_per_epoch=len(train_set),
                                 epochs=epochs,
                                 callbacks=callbacks,
                                 shuffle=False,
                                 **fit_kwargs)

    def predict(self,
                x_data,
                *,
                return_pred_arr = False,
                batch_size: int = 32,
                truncating: bool = False,
                multi_label_threshold: float = 0.5,
                predict_kwargs: Dict = None) -> Union[ClassificationLabelVar, MultiLabelClassificationLabelVar]:
        """
        Generates output predictions for the input samples.

        Computation is done in batches.

        Args:
            x_data: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
            batch_size: Integer. If unspecified, it will default to 32.
            truncating: remove values from sequences larger than `model.embedding.sequence_length`
            multi_label_threshold:
            predict_kwargs: arguments passed to ``predict()`` function of ``tf.keras.Model``

        Returns:
            array(s) of predictions.
        """
        if predict_kwargs is None:
            predict_kwargs = {}
        with kashgari.utils.custom_object_scope():
            if truncating:
                seq_length = self.sequence_length
            else:
                seq_length = None
            tensor = self.text_processor.transform(x_data[0],
                                                   segment=self.embedding.segment,
                                                   seq_length=seq_length,
                                                   max_position=self.embedding.max_position)

            features = np.array(x_data[1])

            logger.debug(f'predict input shape {np.array(tensor).shape}')
            pred = self.tf_model.predict([tensor, features], batch_size=batch_size, **predict_kwargs)

            for i in range(self.task_num):
                logger.debug(f'predict output{i} shape {pred[i].shape}')
            if self.multi_label:
                multi_label_binarizer = self.label_processor.multi_label_binarizer  # type: ignore
                res = multi_label_binarizer.inverse_transform(pred[0],
                                                              threshold=multi_label_threshold)
            else:
                res = []
                for i in range(self.task_num):
                    pred_argmax = pred[i].argmax(-1)
                    lengths = [len(sen) for sen in x_data]
                    if self.task_num == 1:
                        res.append(self.label_processor.inverse_transform(pred.argmax(-1),
                                                                          lengths=lengths))
                        break
                    # res.append(self.label_processor[i].inverse_transform(pred_argmax,
                    #                                                  lengths=lengths))

                    # 因为之前利用了multi_label的label_processor做one-hot输入，详见self.label_processor初始化部分
                    res.append([self.label_processor[i].idx2vocab[label] for label in pred_argmax])
        if return_pred_arr:
            return res, pred
        return res

    def evaluate(self,  # type: ignore[override]
                 x_data: TextSamplesVar,
                 y_data: Union[ClassificationLabelVar, MultiLabelClassificationLabelVar],
                 *,
                 batch_size: int = 32,
                 digits: int = 4,
                 multi_label_threshold: float = 0.5,
                 truncating: bool = False, ) -> Dict:
        y_pred = self.predict(x_data,
                              batch_size=batch_size,
                              truncating=truncating,
                              multi_label_threshold=multi_label_threshold)

        if self.multi_label:
            report = multi_label_classification_report(y_data,  # type: ignore
                                                       y_pred,  # type: ignore
                                                       binarizer=self.label_processor.multi_label_binarizer)  # type: ignore

        else:
            report = []
            # 适配多任务
            for i in range(self.task_num):
                original_report = sklearn_metrics.classification_report(y_data[i],
                                                                        y_pred[i],
                                                                        output_dict=True,
                                                                        digits=digits)
                print(sklearn_metrics.classification_report(y_data[i],
                                                            y_pred[i],
                                                            output_dict=False,
                                                            digits=digits))
                print("saving counfusion matrix...")
                pickle.dump(sklearn_metrics.confusion_matrix(y_data[i], y_pred[i]),
                            open('./confusion_matrix.pkl', 'wb'))

                report.append({
                    'detail': original_report,
                    **original_report['weighted avg']
                })

        return report


if __name__ == "__main__":
    pass
