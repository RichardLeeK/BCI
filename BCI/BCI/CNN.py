
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution1D, Dropout, Conv2D, MaxPooling2D

def create_model(input_dim=32):
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(input_dim, input_dim, 3)))
  model.add(MaxPooling2D(pool_size=(3, 3)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(3, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

