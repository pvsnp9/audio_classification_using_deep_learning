#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:18:48 2020

@author: tsuyogbasnet
"""

import os
import pickle
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

def build_predictions(audio_dir):
    y_true = []
    y_pred = []
    fn_prob = {}
    
    print("Extracting feature from audio files")
    for file in tqdm(os.listdir(audio_dir)):
        rate, signal = wavfile.read(os.path.join(audio_dir, file))
        label = filename_to_class[file]
        c = classes.index(label)
        y_prob = []
        
        for i in range(0, signal.shape[0]-config.step, config.step):
            sample = signal[i:i+config.step]
            x = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
            x = (x-config._min) / (config._max - config._min)
            
            if config.mode == 'conv':
                x = x.reshape(1, x.shape[0], x.shape[1], 1)
            elif config.mode == 'time':
                x = np.expand_dims(x, axis=0)
            
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(c)
        
        fn_prob[file] = np.mean(y_prob, axis=0).flatten()
        
    return y_true, y_pred, fn_prob
    
    
data_frame = pd.read_csv('instruments.csv')
classes = list(np.unique(data_frame.label))
filename_to_class = dict(zip(data_frame.fname,data_frame.label))

p_path = os.path.join('pickles','conv.p')
with open(p_path, 'rb') as handle:
    config = pickle.load(handle)
    
model = load_model(config.model_path)
y_true, y_pred, fn_prob = build_predictions('cleanfiles')

acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)
y_probs = []

for i, row in data_frame.iterrows():
    y_prob = fn_prob[row.fname]
    y_probs.append(y_prob)
    
    for c, p in zip(classes, y_prob):
        data_frame.at[i,c] = p
        

y_pred = [classes[np.argmax(y)] for y in y_probs]
data_frame['y_pred'] = y_pred        

data_frame.to_csv('prediction.csv', index=False)







