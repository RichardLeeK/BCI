from keras.layers.core import Dense, Permute, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.regularizers import l2

def create_model(input_shape=(6, 6, 5)):
    weight_decay = 0.001
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, W_regularizer=l2(weight_decay), activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    '''
    model.add(Convolution2D(128, 3, 3, W_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(256, 3, 3, W_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, 3, 3, W_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, 3, 3, W_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, 3, 3, W_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, 3, 3, W_regularizer=l2(weight_decay), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    '''

    # (bs, y, x, c) --> (bs, x, y, c)
    model.add(Permute((2, 1, 3)))

    # (bs, x, y, c) --> (bs, x, y * c)
    bs, x, y, c = model.layers[-1].output_shape
    model.add(Reshape((x, y*c)))

    model.add(Bidirectional(LSTM(256, return_sequences=False), merge_mode="concat"))
    model.add(Dense(3, activation="softmax"))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adadelta', metrics = ['accuracy'])
    return model