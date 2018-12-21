
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution1D, Dropout, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, Activation, Input, UpSampling3D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from sklearn.metrics import cohen_kappa_score
from keras import backend as K
import numpy as np
from keras.losses import categorical_crossentropy
from keras.models import Model

def create_model(input_dim=(32, 32, 3)):
  model = Sequential()
  model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=input_dim))
#  model.add(MaxPooling2D(pool_size=(3, 3)))
  model.add(Conv2D(6, (3, 3), activation='relu'))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(64, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(3, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
  print(model.summary())
  return model


def cohen_loss(y_true, y_pred):
  return K.sum(K.log(y_true) - K.log(y_pred))


def create_3d_model(input_dim=(9, 18, 5, 1)):
  model = Sequential()
  model.add(Conv3D(32, kernel_size=(3, 1, 1), input_shape=(input_dim), border_mode='same'))
  model.add(Activation('relu'))
  model.add(Conv3D(32, kernel_size=(1, 6, 1), border_mode='same'))
  model.add(Activation('relu'))
  model.add(Conv3D(32, kernel_size=(1, 1, 2), border_mode='same'))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  #model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='valid'))
  #model.add(Activation('relu'))
  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(3, activation='softmax'))
  print(model.summary())
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model


