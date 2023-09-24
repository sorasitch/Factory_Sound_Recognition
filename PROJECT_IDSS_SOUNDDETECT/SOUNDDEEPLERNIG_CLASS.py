# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:11:44 2020

@author: choms
"""

from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
#from statsmodels.api import datasets
from sklearn import datasets ## Get dataset from sklearn
import sklearn.model_selection as ms
import sklearn.metrics as sklm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.random as nr
import sys
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load


# deep learning

import numpy.random as nr
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
import keras.utils.np_utils as ku
import keras.models as models
import keras.layers as layers
from keras import regularizers
from keras.layers import Dropout , Reshape
from keras.optimizers import rmsprop, Adam


from keras import optimizers
import numpy as np
import numpy.linalg as nll
import sklearn.model_selection as ms
import time
import matplotlib.pyplot as plt1
import math

#from keras import layers
from keras.datasets import imdb
from keras import preprocessing as prepro
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, SimpleRNN, LSTM, GRU, Bidirectional
from keras import regularizers
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import numpy.random as nr
import sys

from numpy.random import randint
from numpy import argmax

import warnings
from keras.callbacks import Callback

import os.path
import time


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
            
        if (self.monitor=='val_loss') :
            pass
            if current < self.value :
                if self.verbose > 0:
                    print("Epoch %05d: early stopping THR" % epoch)
                self.model.stop_training = True 
                
        elif (self.monitor=='val_acc') :
            pass
            if current > self.value :
                if self.verbose > 0:
                    print("Epoch %05d: early stopping THR" % epoch)  
                self.model.stop_training = True 
                
        else :
            pass
            if current < self.value :
                if self.verbose > 0:
                    print("Epoch %05d: early stopping THR" % epoch)
                self.model.stop_training = True 

class SoundDeepLerning_Class():
    
    
    def __init__(self
                 ):
        self.X_train = []
        self.Y_train = []
        self.ModelName_train = []
        self.Owners_load = []
        self.Sounds_load = []
        self.ModelName_load = []
        self.Label_load = []
        self.fig, self.axs = plt.subplots(2, 1)
        self.ScaleObj = []
        self.DLObj = []
      
## Deep Learning

    def score_modelDL(self, probs, threshold):
        return np.array([1 if x > threshold else 0 for x in probs[:]])

    def print_metricsDL(self, labels, probs, threshold):
        scores = self.score_modelDL(probs, threshold)
        metrics = sklm.precision_recall_fscore_support(labels, scores)
        conf = sklm.confusion_matrix(labels, scores)
        print('                 Confusion matrix')
        print('                 Score positive    Score negative')
        print('Actual positive 0    %6d' % conf[0,0] + '             %5d' % conf[0,1])
        print('Actual negative 1   %6d' % conf[1,0] + '             %5d' % conf[1,1])
        print('')
        print('Accuracy        %0.2f' % sklm.accuracy_score(labels, scores))
        print('AUC             %0.2f' % sklm.roc_auc_score(labels, probs[:]))
        print('Macro precision %0.2f' % float((float(metrics[0][0]) + float(metrics[0][1]))/2.0))
        print('Macro recall    %0.2f' % float((float(metrics[1][0]) + float(metrics[1][1]))/2.0))
        print(' ')
        print('           Positive      Negative')
        print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
        print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
        print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
        print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])
        return scores

    def score_modelMultiClassifierDL(self, probs, threshold):
        return np.array([1 if x > threshold else 0 for x in probs[:]])

    def print_metricsMultiClassifierDL(self, DLlabels, probs, threshold):
        labels=np.array([ argmax(DLlabels[x]) for x in range(DLlabels.shape[0])])
        print(labels)
        print(probs)
        scores = np.array([ argmax(self.score_modelMultiClassifierDL(probs[x],threshold)) for x in range(probs.shape[0])]) 
        metrics = sklm.precision_recall_fscore_support(labels, scores)
        conf = sklm.confusion_matrix(labels, scores)
        print('                 Confusion matrix')    
        print('Y-Actual X-ScorePredict')
        print(conf)
        print('')
        print('Accuracy        %0.2f' % sklm.accuracy_score(labels, scores))
        print(' ')
        print("Num case [3]   : ",metrics[3])
        print("Precision [0]  : ",metrics[0])
        print("Recall [1]     : ",metrics[1])
        print("F1 [2]         : ",metrics[2])
        return scores 
    
    def plot_loss(self, history):
    #    Function to plot the loss vs. epoch
        train_loss = history.history['loss']
        test_loss = history.history['val_loss']
        x = list(range(1, len(test_loss) + 1))
        self.axs[0].plot(x, test_loss, color = 'red', label = 'Test loss')
        self.axs[0].plot(x, train_loss, label = 'Training losss')
        self.axs[0].legend()
        self.axs[0].set_xlabel('Epoch')
        self.axs[0].set_ylabel('Loss')
        self.axs[0].set_title('Loss vs. Epoch')

    
    def plot_accuracy(self, history):
        train_acc = history.history['acc']
        test_acc = history.history['val_acc']
        x = list(range(1, len(test_acc) + 1))
        self.axs[1].plot(x, test_acc, color = 'red', label = 'Test accuracy')
        self.axs[1].plot(x, train_acc, label = 'Training accuracy')
        self.axs[1].legend()
        self.axs[1].set_xlabel('Epoch')
        self.axs[1].set_ylabel('Accuracy')
        self.axs[1].set_title('Accuracy vs. Epoch')    


    def TrainModel(self
                 ):
        pass
        bidirection = Sequential()
        numofdata=self.X_train.shape[0]
        sizey=self.X_train.shape[1]
        sizex=self.X_train.shape[2]
        category=self.Y_train.shape[1]
        

        scale = preprocessing.StandardScaler()
        x1=self.X_train.reshape((numofdata, sizex * sizey)).astype('float')
        scale.fit(x1)
        dump(scale,self.ModelName_train + ".joblib")     
        x2=scale.transform(x1)
        x3=x2.reshape((numofdata, sizex , sizey)).astype('float')


        INPUT_SHAPE = (sizex,sizey,1)
        x_train = x3.reshape((numofdata, sizex, sizey, 1)).astype('float')
        y_train = self.Y_train
        modelname=self.ModelName_train+".hdf51"
        
#        x_train = self.X_train.reshape((numofdata, sizex * sizey)).astype('float')
#        bidirection.add(Dense(512, activation = 'relu',input_shape=(sizex * sizey,))) 
        
        
        bidirection.add(layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = INPUT_SHAPE))
        bidirection.add(layers.MaxPooling2D(2, 2))
#        bidirection.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
#        bidirection.add(layers.MaxPooling2D(2, 2))
#        bidirection.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
        bidirection.add(layers.Flatten())

  
#        bidirection.add(layers.Conv1D(64, 9, activation='relu', input_shape=(784, 1)))
#        bidirection.add(layers.Conv1D(64, 9, activation='relu'))
#        bidirection.add(layers.MaxPooling1D(3))
#        bidirection.add(layers.Conv1D(64, 9, activation='relu'))
#        bidirection.add(layers.Conv1D(64, 9, activation='relu'))
#        bidirection.add(layers.GlobalAveragePooling1D())
#        bidirection.add(Dropout(0.5))
    
    
#        max_features = 1000
#        max_len = 250
        
        
#        bidirection.add(Embedding(max_features, 32, embeddings_regularizer = regularizers.l2(0.01)))
#        bidirection.add(GRU(32, kernel_regularizer = regularizers.l2(0.01)))
 
       
#        bidirection.add(Reshape(26, 1))
#        RNN = layers.LSTM
#        bidirection.add(RNN(128, input_shape=(26,1)))
#        bidirection.add(layers.Flatten())
#        bidirection.add(RNN(128))
        
        
#        bidirection.add(GRU(32, dropout=0.5, recurrent_dropout=0.1,input_shape=(26,1)))
        
     
#        bidirection.add(Embedding(max_features, 32, embeddings_regularizer = regularizers.l2(0.01)))
#        bidirection.add(Bidirectional(GRU(32, dropout=0.3, recurrent_dropout=0.01))) 
        
        
#        bidirection.add(Embedding(max_features, 32, embeddings_regularizer = regularizers.l2(0.01)))
#        bidirection.add(LSTM(32, return_sequences=True,kernel_regularizer = regularizers.l2(0.01)))
#        bidirection.add(LSTM(32))
        
        
        
#        bidirection.add(Embedding(max_features, 32, embeddings_regularizer = regularizers.l2(0.01)))
#        bidirection.add(Bidirectional(LSTM(32, return_sequences=True,kernel_regularizer = regularizers.l2(0.01))))
#        bidirection.add(Bidirectional(LSTM(32)))
        
        
#        bidirection.add(Dense(32, activation = 'relu',
#                                kernel_regularizer=regularizers.l2(0.01)))
#        bidirection.add(Dropout(0.5))
#        bidirection.add(BatchNormalization(momentum = 0.99))
        
        
#        Best pratices
#        bidirection.add(LSTM(128, return_sequences=True,input_shape=(26,1 )))
#        bidirection.add(LSTM(64, return_sequences=True,input_shape=(26,1 )))
#        bidirection.add(LSTM(32, return_sequences=True,input_shape=(26,1 )))
#        bidirection.add(LSTM(16, return_sequences=True,input_shape=(26,1 )))
#        bidirection.add(layers.Flatten())
        
        
#        bidirection.add(Bidirectional(LSTM(128, return_sequences=True),input_shape=(26,1 )))
#        bidirection.add(Bidirectional(LSTM(64, return_sequences=True),input_shape=(26,1 )))
#        bidirection.add(Bidirectional(LSTM(32, return_sequences=True),input_shape=(26,1 )))
#        bidirection.add(Bidirectional(LSTM(16, return_sequences=True),input_shape=(26,1 )))
#        bidirection.add(layers.Flatten())
        
        
#        bidirection.add(Dense(16, activation = 'relu'))
#                                kernel_regularizer=regularizers.l2(0.01)))
#        bidirection.add(Dropout(0.5))
#        bidirection.add(BatchNormalization(momentum = 0.99))
        
        
#        bidirection.add(Dense(512, activation = 'relu',
#                                kernel_regularizer=regularizers.l2(0.01),input_shape=(26,)))
#        bidirection.add(Dropout(0.5))
#        bidirection.add(BatchNormalization(momentum = 0.99))
#        bidirection.add(Dense(256, activation = 'relu',
#                                kernel_regularizer=regularizers.l2(0.01)))
#        bidirection.add(Dropout(0.5))
#        bidirection.add(BatchNormalization(momentum = 0.99))
#        bidirection.add(Dense(64, activation = 'relu',
#                                kernel_regularizer=regularizers.l2(0.01)))
#        bidirection.add(Dropout(0.5))
#        bidirection.add(BatchNormalization(momentum = 0.99))
#        bidirection.add(Dense(32, activation = 'relu',
#                                kernel_regularizer=regularizers.l2(0.01)))
#        bidirection.add(Dropout(0.5))
#        bidirection.add(BatchNormalization(momentum = 0.99))
        

#        bidirection.add(Bidirectional(LSTM(256, return_sequences=True),input_shape=(2,13 )))
#        bidirection.add(Dropout(0.5))
##        bidirection.add(BatchNormalization(momentum = 0.99))
#        
#        bidirection.add(Bidirectional(LSTM(256, return_sequences=True),input_shape=(2,13 )))
#        bidirection.add(Dropout(0.5))
##        bidirection.add(BatchNormalization(momentum = 0.99))
#        
#        bidirection.add(Bidirectional(LSTM(128, return_sequences=True),input_shape=(2,13 )))
#        bidirection.add(Dropout(0.5))
##        bidirection.add(BatchNormalization(momentum = 0.99))
#        
#        bidirection.add(Bidirectional(LSTM(128, return_sequences=True),input_shape=(2,13 )))
#        bidirection.add(Dropout(0.5))
##        bidirection.add(BatchNormalization(momentum = 0.99))
#        
#        bidirection.add(Bidirectional(LSTM(64, return_sequences=True),input_shape=(2,13 )))
#        bidirection.add(Dropout(0.5))
##        bidirection.add(BatchNormalization(momentum = 0.99))
#        
#        bidirection.add(Bidirectional(LSTM(32, return_sequences=True),input_shape=(2,13 )))
#        bidirection.add(Dropout(0.5))
##        bidirection.add(BatchNormalization(momentum = 0.99))
  
    
#        bidirection.add(Bidirectional(LSTM(256, return_sequences=True),input_shape=(26,1 )))
#        bidirection.add(Dropout(0.5))
##        bidirection.add(BatchNormalization(momentum = 0.99))
#        
#        bidirection.add(Bidirectional(LSTM(256, return_sequences=True),input_shape=(26,1 )))
#        bidirection.add(Dropout(0.5))
##        bidirection.add(BatchNormalization(momentum = 0.99))
#        
#        bidirection.add(Bidirectional(LSTM(128, return_sequences=True),input_shape=(26,1 )))
#        bidirection.add(Dropout(0.5))
##        bidirection.add(BatchNormalization(momentum = 0.99))
#        
#        bidirection.add(Bidirectional(LSTM(128, return_sequences=True),input_shape=(26,1 )))
#        bidirection.add(Dropout(0.5))
##        bidirection.add(BatchNormalization(momentum = 0.99))
#        
#        bidirection.add(Bidirectional(LSTM(64, return_sequences=True),input_shape=(26,1 )))
#        bidirection.add(Dropout(0.5))
##        bidirection.add(BatchNormalization(momentum = 0.99))
#        
#        bidirection.add(Bidirectional(LSTM(32, return_sequences=True),input_shape=(26,1 )))
#        bidirection.add(Dropout(0.5))
##        bidirection.add(BatchNormalization(momentum = 0.99))
#        
#        bidirection.add(layers.Flatten())
        
          
#        bidirection.add(Dense(512, activation = 'relu',input_shape=(26,)))  
#        bidirection.add(Dense(512, activation = 'relu',input_shape=(26,)))
#        bidirection.add(Dense(256, activation = 'relu',input_shape=(26,))) 
#        bidirection.add(Dense(256, activation = 'relu',input_shape=(26,))) 
#        bidirection.add(Dense(128, activation = 'relu',input_shape=(26,))) 
#        bidirection.add(Dense(128, activation = 'relu',input_shape=(26,)))  
#        bidirection.add(Dense(64, activation = 'relu',input_shape=(26,))) 
#        bidirection.add(Dense(32, activation = 'relu',input_shape=(26,)))  
#        bidirection.add(Dense(16, activation = 'relu',input_shape=(26,))) 
        
#        bidirection.add(Dense(64, activation = 'relu')) 
#        bidirection.add(Dense(32, activation = 'relu'))  
#        bidirection.add(Dense(16, activation = 'relu')) 
        #   
   
#        bidirection.add(Dense(1024, activation = 'relu'))  
#        bidirection.add(Dense(1024, activation = 'relu')) 
#        bidirection.add(Dense(512, activation = 'relu'))  
        bidirection.add(Dense(512, activation = 'relu'))
#        bidirection.add(Dense(256, activation = 'relu')) 
        bidirection.add(Dense(256, activation = 'relu')) 
#        bidirection.add(Dense(128, activation = 'relu')) 
        bidirection.add(Dense(128, activation = 'relu'))  
        bidirection.add(Dense(64, activation = 'relu')) 
        bidirection.add(Dense(32, activation = 'relu'))  
        bidirection.add(Dense(16, activation = 'relu')) 
        
        
        SGD = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.005, nesterov=False)
        ADAM = Adam(decay = 0.005)
        RMS = optimizers.RMSprop(lr=0.01)
        RMSprop = 'RMSprop'
        ## Compile the model
        bidirection.add(Dense(int(category), activation = 'softmax'))
        bidirection.compile(optimizer = ADAM, loss = 'categorical_crossentropy', metrics = ['acc'])
        bidirection.summary()
        
        #sys.exit(0)
        
        ## Set up and call-backs for early stopping
        
        callbacks_list = [
        #        keras.callbacks.EarlyStopping(
        #            monitor = 'val_loss', # Use accuracy to monitor the model
        #            min_delta = 0.035, #: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
        #            patience = 1 # Stop after one step with lower accuracy
        #            baseline = 0.035 #: Baseline value for the monitored quantity to reach. Training will stop if the model doesn't show improvement over the baseline.
        #        ),
#                    stopAtLossValue(),
#                EarlyStoppingByLossVal(monitor='val_acc', value=0.998, verbose=1),
                EarlyStoppingByLossVal(monitor='val_loss', value=0.005, verbose=1),
                keras.callbacks.ModelCheckpoint(
                    filepath = modelname, # file where the checkpoint is saved
                    monitor = 'val_loss', # Don't overwrite the saved model unless val_loss is worse
                    save_best_only = True # Only save model if it is the best
                )
        ]
        
        nr.seed(8383)
        
        historyBi = bidirection.fit(x_train, y_train,
                           epochs = 3000,
#                           batch_size = 1024,
                           validation_data = (x_train, y_train)
                           ,callbacks = callbacks_list  # Call backs argument here
                           ,verbose = 1     
                           )
        
        
        self.plot_loss(historyBi)   
        
        self.plot_accuracy(historyBi)
        
        plt.show()
        
        DLmodel = keras.models.load_model(modelname,compile=False)
        predictions = DLmodel.predict(x_train)
        self.print_metricsMultiClassifierDL(y_train,predictions,0.9)

    def TestModel(self,
                  modelname,
                  x,
                  y,
                 ):
        pass
    
        numofdata=x.shape[0]
        sizey=x.shape[1]
        sizex=x.shape[2]
        
        
        scale = load(modelname + ".joblib") 
        x1=x.reshape((numofdata, sizex * sizey)).astype('float')
        x2=scale.transform(x1)
        x3=x2.reshape((numofdata, sizex , sizey)).astype('float')
        
        x4 = x3.reshape((numofdata, sizex, sizey, 1)).astype('float')
#        x1 = x.reshape((numofdata, sizex * sizey)).astype('float')
    
        DLmodel = keras.models.load_model(modelname+".hdf51",compile=False)
        predictions = DLmodel.predict(x4)
        self.print_metricsMultiClassifierDL(y,predictions,0.5)
    
    def LoadModel(self
                 ):
        pass
        self.ScaleObj=load(self.ModelName_load + ".joblib") 
        self.DLObj= keras.models.load_model(self.ModelName_load+".hdf51",compile=False)
    

    def TestLoadModel(self,
                  x,
                  thd,
                 ):
        pass
    
        numofdata=x.shape[0]
        sizey=x.shape[1]
        sizex=x.shape[2]
        
        
        x1=x.reshape((numofdata, sizex * sizey)).astype('float')
        x2=self.ScaleObj.transform(x1)
        x3=x2.reshape((numofdata, sizex , sizey)).astype('float')
        
        x4 = x3.reshape((numofdata, sizex, sizey, 1)).astype('float')
#        x1 = x.reshape((numofdata, sizex * sizey)).astype('float')
        predictions = self.DLObj.predict(x4)
        
        label= np.array([ argmax(self.score_modelMultiClassifierDL(predictions[x],thd)) for x in range(predictions.shape[0])]) 
#        print(predictions)
#        print(label)
        
        return label , predictions[0,int(label)]
