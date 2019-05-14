##import 
from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping,  History
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.style.use('ggplot')

from keras.datasets import imdb


# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2


print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

###2- Pad sequence 
print('Pad sequences (samples x time)')


##Pad 
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)







print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions (a dence vector of type word2vec float) 
##features extractions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
# Regularization of the embedding layer 
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))

# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a Relu hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))


# We project into a single unit output layer, and squash it with a sigmoid activation function (2 classes):
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compile : error and optimizer 
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

##callbacks
#SAve the model
checkpointer = ModelCheckpoint(filepath="model_imbd.hdf5", 
                               verbose=1, save_best_only=True)

#Stop the training
earlystop = EarlyStopping(monitor = 'val_loss', patience=2)


##Model summary 
print(model.summary())

 
# Training the model 
time_start = time.time()

hist_model_imbd = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs, validation_data=(x_test, y_test),
callbacks = [checkpointer,earlystop])
time_end = time.time()

D = time_end  - time_start


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


##Plot the history opf the imbd model 
plt.clf()
plot_history(hist_model_imbd)
plt.savefig('training_and_validation_conv_1D_minst.png', bbox_inches='tight')
#plt.show()  ## You can try this one instead if you are in an interactive session
