from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping,  History
import time
from keras import backend as K
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.style.use('ggplot')


batch_size = 128
num_classes = 10
epochs = 2

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


###Define the input shape of the first convolution layer 
##Need to distinguish the channel as in theano the data are in channel first while they are in channel last in tensorflow   
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


##set type as float for normalizarion (casting) 
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


##Normalization: network more functionnal using  normalized data 
x_train /= 255
x_test /= 255

##visiualisation
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# convert class vectors to binary class matrices (num_classes = 10 as numbers from 0 to 9)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


##Model architecture 
model = Sequential()

#First convolution layer, must specify input_shape as convolution is the first layer 
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

##Second convolution layer 
model.add(Conv2D(64, (3, 3), activation='relu'))

#Maxpooling with a maximum 
model.add(MaxPooling2D(pool_size=(2, 2)))
##Regularization 
model.add(Dropout(0.25))

##layer that changes the dimension of the output (from several list to one row list) 
model.add(Flatten())

##Non-linear dense layer
model.add(Dense(128, activation='relu'))

##Regularization 
model.add(Dropout(0.5))

###Output layer with softmax activation (generalization of logistic with more than 2 classes)
model.add(Dense(num_classes))
model.add(Activation('softmax'))

##compile 
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

##callbacks
checkpointer = ModelCheckpoint(filepath="model_mnist.hdf5", 
                               verbose=1, save_best_only=True)

earlystop = EarlyStopping(monitor = 'val_loss', patience=3)

#Model summary 
print(model.summary())

###training 
time_start = time.time()

hist_model_mnist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test)
,callbacks = [checkpointer,earlystop])

time_end = time.time()

D = time_end  - time_start


##Model evaluation 
score = model.evaluate(x_test, y_test, verbose=0)


####A function that can be used to plot training and validation error 
def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()



###Training validation plot using the defined function
plt.clf()
plot_history(hist_model_mnist)
plt.savefig('training_and_validation_conv_2D_minst.png', bbox_inches='tight')





