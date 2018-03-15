from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, MaxPool2D
from keras.layers.convolutional import Conv2D
from keras.layers import Input, add
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU


def RCL_block(l_settings, l, pool=True):
    input_num_filters = l_settings.output_shape[1]
    conv1 = Conv2D(input_num_filters, (1, 1), padding='same')
    stack1 = conv1(l)   	
    stack2 = BatchNormalization()(stack1)
    stack3 = PReLU()(stack2)

    conv2 = Conv2D(input_num_filters, (3, 3), padding='same', kernel_initializer = 'he_normal')
    stack4 = conv2(stack3)
    #stack5 = merge([stack1, stack4], mode='sum')
    stack5 = add([stack1, stack4])
    stack6 = BatchNormalization()(stack5)
    stack7 = PReLU()(stack6)

    conv3 = Conv2D(input_num_filters, (3, 3), padding='same', weights = conv2.get_weights())
    stack8 = conv3(stack7)
    #stack9 = merge([stack1, stack8], mode='sum')
    stack9 = add([stack1, stack8])
    stack10 = BatchNormalization()(stack9)
    stack11 = PReLU()(stack10)    

    conv4 = Conv2D(input_num_filters, (3, 3), padding='same', weights = conv2.get_weights())
    stack12 = conv4(stack11)
    #stack13 = merge([stack1, stack12], mode='sum')
    stack13 = add([stack1, stack12])
    stack14 = BatchNormalization()(stack13)
    stack15 = PReLU()(stack14)
    if pool:
      stack15 = MaxPool2D((2, 2), padding='same')(stack15)
    #stack16 = stack15
    stack16 = Dropout(0.4)(stack15)
            
    return stack16


def create_model(csp_val=3, output_dim=3, nb_layer=2):
  input = Input(shape=(4, 4, 4))
  conv = Conv2D(128, (3, 3), padding='same', activation='relu')
  l = conv(input)

  for i in range(nb_layer):
    if i % 2 == 0:
      l = RCL_block(conv, l, pool=False)
    else:
      l = RCL_block(conv, l, pool=True)
    print(i)
  out = Flatten()(l)
  out = Dense(3, activation='softmax')(out)
  model = Model(input = input, output = out)
  model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
  return model

