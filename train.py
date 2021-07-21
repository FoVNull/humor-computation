# Multimodality
import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import jieba
from sklearn.model_selection import train_test_split
import sys

from models.RCNN_Text_Img import RCNN_Text_Img
from kashgari.embeddings import BertEmbedding
from transformers import BertTokenizer
from models.kashigari_local.hfbert_embedding import HFBertEmbedding
from kashgari.tasks.classification import BiLSTM_Model


def read_train_data(path, pure=None):
    df = pd.read_excel(path+'/train_data.xlsx', header=0)
    x = []
    img_path = []
    y = []
    for i in range(len(df)):
        img_path.append(path+'/image/' + str(i+1) + '.jpg')
        x.append(
            [word for word in tokenizer.tokenize(df['text'][i])]
        )
        if pure:
            y.append(df['label'][i])
        else:
            y.append([df['label'][i]])

    return x, y, img_path


def read_task_data(task_num):
    test_data = pd.read_excel(prefix+'/datasets/test_data/task_'+str(task_num)+'/test_data.xlsx')
    x = []
    img_path = []
    for i in range(len(test_data)):
        if task_num == 1:
            img_path.append(prefix+'/datasets/test_data/task_'+str(task_num)+'/image/'+str(i+1)+'_a.jpg')
            img_path.append(prefix+'/datasets/test_data/task_'+str(task_num)+'/image/'+str(i+1)+'_b.jpg')
            x.append(
                [word for word in tokenizer.tokenize(test_data['text_a'][i])]
            )
            x.append(
                [word for word in tokenizer.tokenize(test_data['text_b'][i])]
            )
        if task_num == 2:
            img_path.append(prefix+'/datasets/test_data/task_'+str(task_num)+'/image/'+str(i+1)+'.jpg')
            x.append(
                [word for word in tokenizer.tokenize(test_data['text'][i])]
            )
    return x, img_path


def result_gen(task_num, pred, pred_prob=None):
    res = []
    if task_num == 1:
        for i in range(0, len(pred), 2):
            if pred[i] == pred[i+1]:
                res.append(
                    abs(-bool(pred_prob[i][pred[i]-1] > pred_prob[i+1][pred[i]-1]))
                )
            else:
                res.append(
                    abs(-bool(pred[i] > pred[i+1]))
                )
    if task_num == 2:
        res = pred

    with open(prefix+'/datasets/DUFLER_'+str(task_num)+'.csv', 'w', encoding='utf-8') as _f:
        _f.write('id,label\n')
        for idx, _line in enumerate(res):
            _f.write('{},{}\n'.format(idx+1, _line))


def train_only_text():
    x, y, img_paths = read_train_data(prefix+'/datasets/train_data', pure=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # embedding = BertEmbedding('./models/chinese_L-12_H-768_A-12')
    embedding = HFBertEmbedding(prefix+'/models/hfl_chinese_bert_hf')
    models = BiLSTM_Model(embedding)
    models.fit(x_train, y_train, batch_size=32, epochs=10)
    models.evaluate(x_test, y_test)


def img_encode(img_paths):
    img_tensors = []
    for ip in tqdm(img_paths):
        image_raw_data_jpg = tf.io.gfile.GFile(ip, 'rb').read()
        img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg, channels=3)
        img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)

        # 单通道转成3通道
        if tf.shape(img_data_jpg)[2] != 3:
            img_data_jpg = tf.image.grayscale_to_rgb(
                                img_data_jpg,
                                name=None
                            )
        img_tensors.append(tf.image.resize_with_crop_or_pad(img_data_jpg, 224, 224))
    return img_tensors


def train():
    x, y, img_paths = read_train_data(prefix+'/datasets/train_data')
    img_tensors = img_encode(img_paths)

    # embedding = BertEmbedding(prefix+'./models/chinese_L-12_H-768_A-12')
    embedding = HFBertEmbeddingprefix+'/models/hfl_chinese_bert_hf')

    model = RCNN_Text_Img(embedding)

    train_x, test_x, train_img, test_img, trian_y, test_y = train_test_split(x, img_tensors, y, test_size=0.1)
    train_x, vali_x, train_img, vali_img, trian_y, vali_y = train_test_split(train_x, train_img, trian_y, test_size=0.1)
    model.fit(x_train=(train_x, train_img), y_train=trian_y, x_validate=(vali_x, vali_img), y_validate=vali_y,
              epochs=10, batch_size=32, callbacks=None, fit_kwargs=None)

    test_y = list(map(lambda e: list(e), zip(*test_y)))
    reports = model.evaluate((test_x, test_img), test_y, batch_size=64)

    test(model, 1)
    test(model, 2)


def test(model, task_num):
    x, img_paths = read_task_data(task_num=task_num)
    img_tensors = img_encode(img_paths)
    if task_num == 1:
        pred, pred_prob = model.predict((x, img_tensors), batch_size=32, return_pred_arr=True)
        result_gen(task_num, pred[0], pred_prob)
    if task_num == 2:
        pred = model.predict((x, img_tensors), batch_size=32)
        result_gen(task_num, pred[0])


if __name__ == '__main__':
    # 适配datalore的在线路径问题，一般prefix = '.'
    prefix = sys.path[0]
    # stop_words = set()
    # with open(prefix+'/datasets/stop_words.txt', 'r', encoding='utf-8') as f:
    #     for line in f.readlines():
    #         stop_words.add(line.strip())
    tokenizer = BertTokenizer.from_pretrained(prefix+'/models/hfl_chinese_bert_hf')
    with tf.device("/gpu:0"):
        # train_only_text()
        train()
