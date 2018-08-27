
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution1D, Dropout, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, Activation
from keras.layers.advanced_activations import LeakyReLU

def create_model(input_dim=32):
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_dim))
  #model.add(MaxPooling2D(pool_size=(3, 3)))
  #model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(3, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model


def create_3d_model(input_dim=(8, 8, 6)):
  model = Sequential()
  model.add(Conv3D(32, kernel_size=(3, 3, 1), input_shape=(input_dim), border_mode='same'))
  model.add(Activation('relu'))
  model.add(Conv3D(32, kernel_size=(3, 3, 1), border_mode='same'))
  model.add(MaxPooling3D(MaxPooling2D(pool_size=(3, 3, 3)), border_mode='same'))
  model.add(Dropout(0.25))
  model.add(Conv3D(64, kernel_size=(3, 3, 1), border_mode='same'))
  model.add(Activation('relu'))
  model.add(Flatten())
  model.add(Dense(512, activation='sigmoid'))
  model.add(Dropout(0.5))
  model.add(Dense(3, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model


