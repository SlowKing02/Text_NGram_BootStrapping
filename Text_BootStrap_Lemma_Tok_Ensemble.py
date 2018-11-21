#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 17:28:37 2018

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


def load_Texts_sentiment_analysis_dataset():
    split = 0.85
    bs = 1

    Texts_load = pd.read_csv()
    bootstrap = int(bs * len(Texts_load))
    Texts_load_s = Texts_load.sample(n=bootstrap, replace=False)
    length = len(Texts_load_s)
    l = int(length * split)
    Texts_train = Texts_load_s[:l]
    Texts_val = Texts_load_s[l:]
    
    Texts_train_text = list(Texts_train.text)
    Texts_train_labels = np.array(Texts_train.labels)
    Texts_val_text = list(Texts_val.text)
    Texts_val_labels = np.array(Texts_val.labels)
    
    Texts_test_load = pd.read_csv()
    Texts_test_texts = list(Texts_test_load.text)
    
    Texts_id = np.array(Texts_load_s.id)
    Texts_train_predict = pd.read_csv()
    Texts_train_p_list = list(Texts_train_predict.text)

    return ((Texts_train_text, Texts_train_labels), (Texts_val_text, Texts_val_labels), (Texts_test_texts), (Texts_train_p_list))


def ngram_vectorize(train_texts, train_labels, val_texts, test_texts, train_predict):
    """Vectorizes texts as ngram vectors."""
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': (1, 2),  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': 'word',
            'min_df': 2,
            'max_df': .85,
            'token_pattern': r"(?u)\b\w\w+\b|!|\?|\|\'",
    }    
    vectorizer = TfidfVectorizer(**kwargs)
    
    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)

    # Vectorize test texts.
    x_test = vectorizer.transform(test_texts)
    t_predict = vectorizer.transform(train_predict)

    Top_Features = 250000
    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(Top_Features, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train)
    x_val = selector.transform(x_val)
    x_test = selector.transform(x_test)
    t_predict = selector.transform(t_predict)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')
    t_predict = t_predict.astype('float32')
    return x_train, x_val, x_test, t_predict


def mlp_model(layers, units, dropout_rate, input_shape):
    """Creates an instance of a multi-layer perceptron model.
    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
    # Returns
        An MLP model instance.
    """
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=1, activation='sigmoid'))
    return model


def train_ngram_model(data, learning_rate=1e-4, epochs=1000, batch_size=128, layers=3, units=64, dropout_rate=0.35):

    # Get the data.
    (train_texts, train_labels), (val_texts, val_labels), (test_texts), (train_predict) = data
    
    # Vectorize texts.
    x_train, x_val, x_test, t_predict = ngram_vectorize(train_texts, train_labels, val_texts, test_texts, train_predict)

    # Create model instance.
    model = mlp_model(layers=layers, units=units, dropout_rate=dropout_rate, input_shape=x_train.shape[1:])

    # Compile model with learning parameters.
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    # Train and validate model.
    history = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_val, val_labels),
            verbose=2,  # Logs once per epoch.
            batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save('Texts_mlp_model_bs_lemma.h5')
    
    test_predictions = list()
    for each in x_test:
        test_predictions.append(model.predict(each))
        
    train_predictions = list()
    for each in t_predict:
        train_predictions.append(model.predict(each))
        
    return history['val_acc'][-1], history['val_loss'][-1], test_predictions, train_predictions


if __name__ == '__main__':
    # Using the Texts movie reviews dataset to demonstrate training n-gram model
    data = load_Texts_sentiment_analysis_dataset()    
    
    Train_Loc = ['Natural_Train/train_1.csv',
                 'Natural_Train/train_2.csv',
                 'Natural_Train/train_3.csv',
                 'Natural_Train/train_4.csv',
                 'Natural_Train/train_5.csv',
                 'Natural_Train/train_6.csv',
                 'Natural_Train/train_7.csv',
                 'Natural_Train/train_8.csv',
                 'Natural_Train/train_9.csv',
                 'Natural_Train/train_10.csv',
                 'Natural_Train/train_11.csv',
                 'Natural_Train/train_12.csv',
                 'Natural_Train/train_13.csv',
                 'Natural_Train/train_14.csv',
                 'Natural_Train/train_15.csv']
    
    Test_Loc = ['Natural_Train/test_1.csv',
                'Natural_Train/test_2.csv',
                'Natural_Train/test_3.csv',
                'Natural_Train/test_4.csv',
                'Natural_Train/test_5.csv',
                'Natural_Train/test_6.csv',
                'Natural_Train/test_7.csv',
                'Natural_Train/test_8.csv',
                'Natural_Train/test_9.csv',
                'Natural_Train/test_10.csv',
                'Natural_Train/test_11.csv',
                'Natural_Train/test_12.csv',
                'Natural_Train/test_13.csv',
                'Natural_Train/test_14.csv',
                'Natural_Train/test_15.csv']
    
    Labels = ['labels1', 'labels2', 'labels3', 'labels4', 'labels5', 
              'labels6', 'labels7', 'labels8', 'labels9', 'labels10',
              'labels11', 'labels12', 'labels13', 'labels14', 'labels15']

    for i in range(0, len(Labels)):
        (x, y, test_predictions, train_predictions) = train_ngram_model(data)
        Texts_test_load = pd.read_csv('Lemma_Test.csv')
        Texts_test_load[Labels[i]] = np.concatenate(test_predictions, axis=0 )
        Texts_test_load = Texts_test_load.drop(columns = ['text', 'id'])        
        Texts_test_load.to_csv(Test_Loc[i], index=False)
        
        Texts_train = pd.read_csv('Lemma_Train.csv')
        Texts_train[Labels[i]] = np.concatenate(train_predictions, axis=0 )
        Texts_train = Texts_train.drop(columns=['id', 'labels', 'text'])        
        Texts_train.to_csv(Train_Loc[i], index=False)      
    

        
        
    
    
    

    
