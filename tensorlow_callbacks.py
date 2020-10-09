# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 23:18:30 2020

@author: Koray
"""


import tensorflow as tf 
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import fashion_mnist

#Datamızı yüklüyoruz
mnist=fashion_mnist.load_data()

#Mnist datasetini x_train,x_test,y_train,y_test olarak ayırdık
(x_train,y_train),(x_test,y_test)=mnist

#Verimizi 0-1 arasında degerler halıne getırıyoruz

x_train,x_test=x_train/255,x_test/255

#%%

#İstediğimiz Accuracy değerini belirledik burada
ACCURACY_THRESHOLD = 0.85

#Train fonksiyonu oluşturduk.İçinde Sinir ağımız var 
#Aynı zaman da belirlediğimiz accuracy değerinde eğitimin durması için callbacks sınıfı oluşturduk

def train():
    
    class mycallback(Callback):
        
        def on_epoch_end(self,epoch,logs={}):
            
            if logs.get('acc') > 0.85:
                
                print("\nReached %2.2f%% accuracy, so stopping training!" %(ACCURACY_THRESHOLD*100))
                
                self.model.stop_training= True
                
    
    callbacks=mycallback()
    
    model=Sequential([Flatten(input_shape=(28,28)),
                      
                      Dense(units=512,activation=tf.nn.relu),
                      
                      Dense(units=256,activation=tf.nn.relu),
                      
                      Dense(units=10,activation=tf.nn.softmax)
                      ])
    
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    
    history=model.fit(x_train,y_train,epochs=5,callbacks=[callbacks])
        
    return history.epoch,history.history['acc'][-1]




    
    
    
