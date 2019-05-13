#! /bin/bash
## Installation of KERAS in a virtual environment for python
## Workshop Machine Learning 2019 05 17
## Presentation: May Taha
## Assistants: Morgan Soulié Enrique Ortega-Abboud


################################################
## Installation of packages in your system require administrative rights
################################################

## sudo apt-get install python-dev python-tk virtualenv

## To install python 3.5 or 3.6 you should compile it and install it not to
## mess with the system version of python



################################################
## CREATE A VIRTUAL ENVIRONMENT
################################################

## Check if virtualenv is installed, else install it
virtualenv --version

## Create a directory to store your environments
mkdir ~/envs
cd ~/envs

## Find version of python (py 3.5 or 3.6 depending on your system)
which python3.6

## Create the virtual environment with a given version of python
## Use that path in the parameter -p
## name your environment along with the path
virtualenv -p /usr/local/bin/python3.6 ~/envs/pykeras

## Start the environment
source ~/envs/pykeras/bin/activate

## There should be a prefix with the name of the environment before your prompt



################################################
## CHECK PYTHON VERSION AND INSTALL PACKAGES FOR ML IN KERAS
################################################


## It should be the version of python which was given at first
python --version


## Install the dependencies needed for the workshop in the terminal
## the same terminal.
## Virtual environments are only active in one terminal once they are activated
## ** If pip needs to be updated it will give the proper command to do so before installing
pip3 install tensorflow
pip3 install numpy==1.16.1
pip3 install scipy
pip3 install scikit-learn
pip3 install h5py
pip3 install keras
pip3 install pandas
pip3 install matplotlib



## Check if the installation is working by importing keras
## If an error "connection reset by peer" appears,
## you can try to run this chunk of code again (with the proper environment active)
python -c "from __future__ import print_function;
from keras.preprocessing import sequence;
from keras.models import Sequential;
from keras.layers import Dense, Dropout, Flatten, Activation, Embedding;
from keras.layers import Conv1D, GlobalMaxPooling1D;
from keras.layers import Conv2D, MaxPooling2D;
from keras.callbacks import ModelCheckpoint, EarlyStopping,  History;
from keras import backend as K;
from keras.datasets import mnist;
from keras.datasets import imdb;
max_features = 5000 ;
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features) ;
(x2_train, y2_train), (x2_test, y2_test) = mnist.load_data() ;
exit()"


## Once the work is done you can deactivate the environment or close the terminal
deactivate


