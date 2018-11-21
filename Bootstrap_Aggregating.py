#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 00:34:11 2018

@author: slowking
"""
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from tensorflow.python.keras import models

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from keras.models import load_model

Texts_DS = pd.read_csv('train.csv')
Texts_1 = pd.read_csv('Natural_Train/train_1.csv')
Texts_2 = pd.read_csv('Natural_Train/train_2.csv')
Texts_3 = pd.read_csv('Natural_Train/train_3.csv')
Texts_4 = pd.read_csv('Natural_Train/train_4.csv')
Texts_5 = pd.read_csv('Natural_Train/train_5.csv')
Texts_6 = pd.read_csv('Natural_Train/train_6.csv')
Texts_7 = pd.read_csv('Natural_Train/train_7.csv')
Texts_8 = pd.read_csv('Natural_Train/train_8.csv')
Texts_9 = pd.read_csv('Natural_Train/train_9.csv')
Texts_10 = pd.read_csv('Natural_Train/train_10.csv')
Texts_11 = pd.read_csv('Natural_Train/train_11.csv')
Texts_12 = pd.read_csv('Natural_Train/train_12.csv')
Texts_13 = pd.read_csv('Natural_Train/train_13.csv')
Texts_14 = pd.read_csv('Natural_Train/train_14.csv')
Texts_15 = pd.read_csv('Natural_Train/train_15.csv')
split = .6
Texts_Master = pd.concat([Texts_DS, Texts_1, Texts_2, Texts_3, Texts_4, Texts_5, Texts_6, Texts_7, Texts_8, 
                         Texts_9, Texts_10, Texts_11, Texts_12, Texts_13, Texts_14, Texts_15], axis=1, join='inner')
l = int(len(Texts_Master)*split)
training = Texts_Master[:l]
valid = Texts_Master[l:]

x_train = training[['labels1', 'labels2', 'labels3', 'labels4', 'labels5', 'labels6', 'labels7', 'labels8',
                    'labels9', 'labels10', 'labels11', 'labels12', 'labels13', 'labels14', 'labels15']]
train_labels = training['labels']
x_val = valid[['labels1', 'labels2', 'labels3', 'labels4', 'labels5', 'labels6', 'labels7', 'labels8',
                    'labels9', 'labels10', 'labels11', 'labels12', 'labels13', 'labels14', 'labels15']]
val_labels = valid['labels']

model = models.Sequential()
model.add(Dense(units=1, input_dim=15, activation='sigmoid'))

optimizer = tf.keras.optimizers.Adam(lr=1e-4)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    # Train and validate model.
history = model.fit(
            x_train,
            train_labels,
            epochs=25,
            callbacks=callbacks,
            validation_data=(x_val, val_labels),
            verbose=2,  # Logs once per epoch.
            batch_size=128)

    # Print results.
history = history.history
print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

model.save('Texts_mlp_model_bs_ensemble_lemma.h5')

T_1 = pd.read_csv('Natural_Train/test_1.csv')
T_2 = pd.read_csv('Natural_Train/test_2.csv')
T_3 = pd.read_csv('Natural_Train/test_3.csv')
T_4 = pd.read_csv('Natural_Train/test_4.csv')
T_5 = pd.read_csv('Natural_Train/test_5.csv')
T_6 = pd.read_csv('Natural_Train/test_6.csv')
T_7 = pd.read_csv('Natural_Train/test_7.csv')
T_8 = pd.read_csv('Natural_Train/test_8.csv')
T_9 = pd.read_csv('Natural_Train/test_9.csv')
T_10 = pd.read_csv('Natural_Train/test_10.csv')
T_11 = pd.read_csv('Natural_Train/test_11.csv')
T_12 = pd.read_csv('Natural_Train/test_12.csv')
T_13 = pd.read_csv('Natural_Train/test_13.csv')
T_14 = pd.read_csv('Natural_Train/test_14.csv')
T_15 = pd.read_csv('Natural_Train/test_15.csv')

Test_Master = pd.concat([T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10, T_11, T_12, T_13, T_14, T_15], axis=1, join='inner')

output = model.predict(Test_Master)
Texts_test_out = pd.read_csv('Lemma_Test.csv')
Texts_test_out['labels'] = output
Texts_test_out = Texts_test_out.drop(columns='text')
Texts_test_out.to_csv('BootStrap_Natural_Single.csv', index=False)






def Texts_Avg():

    Texts_Avg_1 = pd.read_csv('Natural_Train/test_1.csv')
    Texts_Avg_2 = pd.read_csv('Natural_Train/test_2.csv')
    Texts_Avg_3 = pd.read_csv('Natural_Train/test_3.csv')
    Texts_Avg_4 = pd.read_csv('Natural_Train/test_4.csv')
    Texts_Avg_5 = pd.read_csv('Natural_Train/test_5.csv')
    Texts_Avg_6 = pd.read_csv('Natural_Train/test_6.csv')
    Texts_Avg_7 = pd.read_csv('Natural_Train/test_7.csv')
    Texts_Avg_8 = pd.read_csv('Natural_Train/test_8.csv')
    Texts_Avg_9 = pd.read_csv('Natural_Train/test_9.csv')
    Texts_Avg_10 = pd.read_csv('Natural_Train/test_10.csv')
    Texts_Avg_11 = pd.read_csv('Natural_Train/test_11.csv')
    Texts_Avg_12 = pd.read_csv('Natural_Train/test_12.csv')
    Texts_Avg_13 = pd.read_csv('Natural_Train/test_13.csv')
    Texts_Avg_14 = pd.read_csv('Natural_Train/test_14.csv')
    Texts_Avg_15 = pd.read_csv('Natural_Train/test_15.csv')

    result_avg = pd.concat([Texts_Avg_1, Texts_Avg_2, Texts_Avg_3, Texts_Avg_4, Texts_Avg_5, Texts_Avg_6,
                    Texts_Avg_7, Texts_Avg_8, Texts_Avg_9, Texts_Avg_10, Texts_Avg_11, Texts_Avg_12,
                    Texts_Avg_13, Texts_Avg_14, Texts_Avg_15
                    ], axis=1, join='inner')
    
    result_cnt = pd.concat([Texts_Avg_1, Texts_Avg_2, Texts_Avg_3, Texts_Avg_4, Texts_Avg_5, Texts_Avg_6,
                    Texts_Avg_7, Texts_Avg_8, Texts_Avg_9, Texts_Avg_10, Texts_Avg_11, Texts_Avg_12,
                    Texts_Avg_13, Texts_Avg_14, Texts_Avg_15
                    ], axis=1, join='inner')
    
    Texts_DS = pd.read_csv('test.csv')
    id_col = Texts_DS.id

    result_avg['median'] = result_avg.median(axis=1)
    result_avg['mean'] = result_avg.mean(axis=1)

    result_cnt = result_cnt.round(0)
    result_cnt['vote'] = (result_cnt.sum(axis=1)/15)

    result = result_avg[['median','mean']].copy()
    result['vote'] = result_cnt.vote
    result['id'] = id_col
    result = result[['id','vote','mean','median']]

    top = float(.95)
    bot = float(.05)
    result['optimize'] = np.where((result['vote'] == 1) & (result['mean'] >= top)
      , 0.95, (np.where((result['vote'] == 0) & (result['mean'] <= bot), .05, result['mean'])))
    
    return result

Agg = Texts_Avg()
Agg = Agg.drop(columns=['median', 'vote', 'optimize'])
Agg = Agg.rename(index=str, columns={'mean':'labels'})
Agg.to_csv('Natural_Avg.csv', index=False)

"""Let me try to reduce unsure options, and set ones that are high but low vote to lower values"""

#result = result[['id', 'labels_1', 'labels_2', 'labels_3', 'labels_4', 'labels_5', 'labels_6', 'labels_7', 'labels_8', 'labels_9', 
#                'labels_10' , 'labels_11' , 'labels_12' , 'labels_13' , 'labels_14' , 'labels_15']]



Texts_Avg_1 = pd.read_csv('Natural_Train/test_1_s.csv')
Texts_Avg_1 = Texts_Avg_1.rename(index=str, columns={'labels1':'labels'})
Texts_Avg_1.to_csv('Test_1_ss.csv', index=False)















def get_x_train():
    Texts_load = pd.read_csv('train.csv') 
    train_texts = list(Texts_load.text)
    train_labels = np.array(Texts_load.labels)
    kwargs = {
            'ngram_range': (1, 2),  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': 'word',
            'min_df': 2,
            'max_df': .85,
    }    
    vectorizer = TfidfVectorizer(**kwargs)
    
    x_train = vectorizer.fit_transform(train_texts) 
    Top_Features = 50000
    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(Top_Features, x_train.shape[1]))    
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train)
    x_train = x_train.astype('float32')
    
    model = load_model('Texts_mlp_model_bs_3.h5')

    predictions = list()
    for each in x_train:
        predictions.append(model.predict(each))
        
    return predictions
