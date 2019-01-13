#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
import keras
from keras.utils import Sequence
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Add,Activation,Conv2DTranspose,UpSampling2D, Dropout, Cropping2D,Flatten, Dense, Input, Conv2D, MaxPooling2D, GRU,BatchNormalization,                         Concatenate, LeakyReLU, GlobalAveragePooling2D, Average
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras import metrics
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import backend as K
import keras
import tensorflow as tf
from tensorflow import set_random_seed
from pathlib import Path


# In[2]:


BATCH_SIZE = 128
SEED = 777
SHAPE = (256, 256, 4)
TrainPath = sys.argv[1]
TrainName = sys.argv[2]
TestPath = sys.argv[3]
TestName = sys.argv[4]
VAL_RATIO = 0.1 # 20% as validation
THRESHOLD = 0.05 # due to different cost of True Positive vs False Positive, this is the probability threshold to predict the class as 'yes'

ia.seed(SEED)
set_random_seed(SEED)

# In[3]:


def getTrainDataset():
    
    path_to_train = TrainPath
    data = pd.read_csv(TrainName)

    paths = []
    labels = []
    
    for name, lbl in zip(data['Id'], data['Target'].str.split(' ')):
        y = np.zeros(28)
        for key in lbl:
            y[int(key)] = 1
        paths.append(os.path.join(path_to_train, name))
        labels.append(y)

    return np.array(paths), np.array(labels)

def getTestDataset():
    
    path_to_test = TestPath
    data = pd.read_csv(TestName)

    paths = []
    labels = []
    
    for name in data['Id']:
        y = np.ones(28)
        paths.append(os.path.join(path_to_test, name))
        labels.append(y)

    return np.array(paths), np.array(labels)


# In[4]:


# credits: https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L302
# credits: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class ProteinDataGenerator(keras.utils.Sequence):
            
    def __init__(self, paths, labels, batch_size, shape, shuffle = False, use_cache = False, augment = False):
        self.paths, self.labels = paths, labels
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.augment = augment
        if use_cache == True:
            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]), dtype=np.float16)
            self.is_cached = np.zeros((paths.shape[0]))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx+1) * self.batch_size]

        paths = self.paths[indexes]
        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))
        # Generate data
        if self.use_cache == True:
            X = self.cache[indexes]
            for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
                image = self.__load_image(path)
                self.is_cached[indexes[i]] = 1
                self.cache[indexes[i]] = image
                X[i] = image
        else:
            for i, path in enumerate(paths):
                X[i] = self.__load_image(path)

        y = self.labels[indexes]
                
        if self.augment == True:
            seq = iaa.Sequential([
                iaa.OneOf([
                    iaa.Fliplr(0.5), # horizontal flips
                    iaa.Crop(percent=(0, 0.1)), # random crops
                    # Small gaussian blur with random sigma between 0 and 0.5.
                    # But we only blur about 50% of all images.
                    iaa.Sometimes(0.5,
                        iaa.GaussianBlur(sigma=(0, 0.5))
                    ),
                    # Strengthen or weaken the contrast in each image.
                    iaa.ContrastNormalization((0.75, 1.5)),
                    # Add gaussian noise.
                    # For 50% of all images, we sample the noise once per pixel.
                    # For the other 50% of all images, we sample the noise per pixel AND
                    # channel. This can change the color (not only brightness) of the
                    # pixels.
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5),
                    # Make some images brighter and some darker.
                    # In 20% of all cases, we sample the multiplier once per channel,
                    # which can end up changing the color of the images.
                    iaa.Multiply((0.8, 1.2), per_channel=0.2),
                    # Apply affine transformations to each image.
                    # Scale/zoom them, translate/move them, rotate them and shear them.
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-180, 180),
                        shear=(-8, 8)
                    )
                ])], random_order=True)

            #X = np.concatenate((X, seq.augment_images(X), seq.augment_images(X), seq.augment_images(X)), 0)
            #y = np.concatenate((y, y, y, y), 0)
        
        return X, y
    
    def on_epoch_end(self):
        
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item
            
    def __load_image(self, path):
        R = Image.open(path + '_red.png')
        G = Image.open(path + '_green.png')
        B = Image.open(path + '_blue.png')
        Y = Image.open(path + '_yellow.png')

        im = np.stack((
            np.array(R), 
            np.array(G), 
            np.array(B),
            np.array(Y)), -1)
        
        im = cv2.resize(im, (SHAPE[0], SHAPE[1]))
        im = np.divide(im, 255)
        return im


def f1(y_true, y_pred):
    #y_pred = K.round(y_pred)
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    #y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1-K.mean(f1)


# some basic useless model
def create_model_0(init):
    
    dropRate = 0.25
    
    x = BatchNormalization(axis=-1)(init)
    x = Conv2D(32, (3, 3), activation='relu')(x) #, strides=(2,2))(x)

    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    ginp1 = Dropout(dropRate)(x)
    
    x = BatchNormalization(axis=-1)(ginp1)
    x = Conv2D(64, (3, 3), strides=(2,2), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    ginp2 = Dropout(dropRate)(x)
    
    x = BatchNormalization(axis=-1)(ginp2)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    ginp3 = Dropout(dropRate)(x)
    
    gap1 = GlobalAveragePooling2D()(ginp1)
    gap2 = GlobalAveragePooling2D()(ginp2)
    gap3 = GlobalAveragePooling2D()(ginp3)
    
    x = Concatenate()([gap1, gap2, gap3])
    
    x = BatchNormalization(axis=-1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropRate)(x)
    
    x = BatchNormalization(axis=-1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    
    x = Dense(28)(x)
    x = Activation('sigmoid')(x)
    
    model = Model(init, x)
    
    return model


# In[10]:


# some basic useless model
def create_model_1(init):
    
    dropRate = 0.25
    
    x = BatchNormalization(axis=-1)(init)
    x = Conv2D(16, (5, 5), activation='relu')(x) #, strides=(2,2))(x)

    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    ginp1 = Dropout(dropRate)(x)
    
    x = BatchNormalization(axis=-1)(ginp1)
    x = Conv2D(32, (5, 5), strides=(2,2), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(32, (5, 5), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(32, (5, 5), activation='relu')(x)
    
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    ginp2 = Dropout(dropRate)(x)
    
    x = BatchNormalization(axis=-1)(ginp2)
    x = Conv2D(64, (5, 5), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(64, (5, 5), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(64, (5, 5), activation='relu')(x)
    ginp3 = Dropout(dropRate)(x)
    
    gap1 = GlobalAveragePooling2D()(ginp1)
    gap2 = GlobalAveragePooling2D()(ginp2)
    gap3 = GlobalAveragePooling2D()(ginp3)
    
    x = Concatenate()([gap1, gap2, gap3])
    
    x = BatchNormalization(axis=-1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(2 * dropRate)(x)
    
    x = BatchNormalization(axis=-1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(2 * dropRate)(x)
    
    x = Dense(28)(x)
    x = Activation('sigmoid')(x)
    
    model = Model(init, x)
    
    return model


# In[11]:


# some basic useless model
def create_model_2(init):
    
    dropRate = 0.25
    
    x = BatchNormalization(axis=-1)(init)
    x = Conv2D(32, (2, 2), activation='relu')(x) #, strides=(2,2))(x)

    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    ginp1 = Dropout(dropRate)(x)
    
    x = BatchNormalization(axis=-1)(ginp1)
    x = Conv2D(64, (2, 2), strides=(2,2), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(64, (2, 2), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(64, (2, 2), activation='relu')(x)
    
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    ginp2 = Dropout(dropRate)(x)
    
    x = BatchNormalization(axis=-1)(ginp2)
    x = Conv2D(128, (2, 2), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(128, (2, 2), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(128, (2, 2), activation='relu')(x)
    ginp3 = Dropout(dropRate)(x)
    
    gap1 = GlobalAveragePooling2D()(ginp1)
    gap2 = GlobalAveragePooling2D()(ginp2)
    gap3 = GlobalAveragePooling2D()(ginp3)
    
    x = Concatenate()([gap1, gap2, gap3])
    
    x = BatchNormalization(axis=-1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(2 * dropRate)(x)
    
    x = BatchNormalization(axis=-1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(2 * dropRate)(x)
    
    x = Dense(28)(x)
    x = Activation('sigmoid')(x)
    
    model = Model(init, x)
    
    return model


# In[12]:


# some basic useless model
def create_model_3(init):
    
    dropRate = 0.25
    
    x = BatchNormalization(axis=-1)(init)
    x = Conv2D(32, (2, 2), activation='relu')(x) #, strides=(2,2))(x)

    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    ginp1 = Dropout(dropRate)(x)
    
    x = BatchNormalization(axis=-1)(ginp1)
    x = Conv2D(64, (2, 2), strides=(2,2), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(64, (2, 2), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(64, (2, 2), activation='relu')(x)
    
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    ginp2 = Dropout(dropRate)(x)
    
    x = BatchNormalization(axis=-1)(ginp2)
    x = Conv2D(128, (2, 2), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(128, (2, 2), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(128, (2, 2), activation='relu')(x)
    ginp3 = Dropout(dropRate)(x)
    
    gap1 = GlobalAveragePooling2D()(ginp1)
    gap2 = GlobalAveragePooling2D()(ginp2)
    gap3 = GlobalAveragePooling2D()(ginp3)
    
    x = Concatenate()([gap1, gap2, gap3])
    
    x = BatchNormalization(axis=-1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(2 * dropRate)(x)
    
    x = BatchNormalization(axis=-1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropRate)(x)
    
    x = Dense(28)(x)
    x = Activation('sigmoid')(x)
    
    model = Model(init, x)
    
    return model


# In[13]:


# some basic useless model
def create_model_4(init):
    
    dropRate = 0.25
    
    x = BatchNormalization(axis=-1)(init)
    x = Conv2D(32, (3, 3), activation='relu')(x) #, strides=(2,2))(x)

    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    ginp1 = Dropout(dropRate)(x)
    
    x = BatchNormalization(axis=-1)(ginp1)
    x = Conv2D(64, (3, 3), strides=(2,2), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    ginp2 = Dropout(dropRate)(x)
    
    x = BatchNormalization(axis=-1)(ginp2)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    ginp3 = Dropout(dropRate)(x)
    
    gap1 = GlobalAveragePooling2D()(ginp1)
    gap2 = GlobalAveragePooling2D()(ginp2)
    gap3 = GlobalAveragePooling2D()(ginp3)
    
    x = Concatenate()([gap1, gap2, gap3])
    
    x = BatchNormalization(axis=-1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(2 * dropRate)(x)
    
    x = BatchNormalization(axis=-1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropRate)(x)
    
    x = Dense(28)(x)
    x = Activation('sigmoid')(x)
    
    model = Model(init, x)
    
    return model


# In[14]:


# some basic useless model
def create_model_5(init):
    
    dropRate = 0.25
    
    x = BatchNormalization(axis=-1)(init)
    x = Conv2D(24, (4, 4), activation='relu')(x) #, strides=(2,2))(x)

    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    ginp1 = Dropout(dropRate)(x)
    
    x = BatchNormalization(axis=-1)(ginp1)
    x = Conv2D(48, (4, 4), strides=(2,2), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(48, (4, 4), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(48, (4, 4), activation='relu')(x)
    
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    ginp2 = Dropout(dropRate)(x)
    
    x = BatchNormalization(axis=-1)(ginp2)
    x = Conv2D(96, (4, 4), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(96, (4, 4), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(96, (4, 4), activation='relu')(x)
    ginp3 = Dropout(dropRate)(x)
    
    gap1 = GlobalAveragePooling2D()(ginp1)
    gap2 = GlobalAveragePooling2D()(ginp2)
    gap3 = GlobalAveragePooling2D()(ginp3)
    
    x = Concatenate()([gap1, gap2, gap3])
    
    x = BatchNormalization(axis=-1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(2 * dropRate)(x)
    
    x = BatchNormalization(axis=-1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropRate)(x)
    
    x = Dense(28)(x)
    x = Activation('sigmoid')(x)
    
    model = Model(init, x)
    
    return model


# In[15]:


# some basic useless model
def create_model_6(init):
    
    dropRate = 0.25
    
    x = BatchNormalization(axis=-1)(init)
    x = Conv2D(32, (3, 3), activation='relu')(x) #, strides=(2,2))(x)

    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    ginp1 = Dropout(dropRate)(x)
    
    x = BatchNormalization(axis=-1)(ginp1)
    x = Conv2D(64, (3, 3), strides=(2,2), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    ginp2 = Dropout(dropRate)(x)
    
    x = BatchNormalization(axis=-1)(ginp2)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    ginp3 = Dropout(dropRate)(x)
    
    gap1 = GlobalAveragePooling2D()(ginp1)
    gap2 = GlobalAveragePooling2D()(ginp2)
    gap3 = GlobalAveragePooling2D()(ginp3)
    
    x = Concatenate()([gap1, gap2, gap3])
    
    x = BatchNormalization(axis=-1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropRate)(x)
    
    x = BatchNormalization(axis=-1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    
    x = Dense(28)(x)
    x = Activation('sigmoid')(x)
    
    model = Model(init, x)
    
    return model


# In[16]:


def draw_process(hist, index, phase):
	pass
	'''
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('Model ' + str(index) + ', ' + phase + ': loss')
    ax[0].plot(hist.epoch, hist.history["loss"], label="Train loss")
    ax[0].plot(hist.epoch, hist.history["val_loss"], label="Validation loss")
    ax[1].set_title('Model ' + str(index) + ', ' + phase + ': acc')
    ax[1].plot(hist.epoch, hist.history["f1"], label="Train F1")
    ax[1].plot(hist.epoch, hist.history["val_f1"], label="Validation F1")
    ax[0].legend()
    ax[1].legend()
    plt.savefig('./process/model' + str(index) + '_' + phase + '.png')
	'''

# In[17]:


init = Input(SHAPE)

model = {0: create_model_0(init), 
         1: create_model_1(init), 
         2: create_model_2(init),
         3: create_model_3(init), 
         4: create_model_4(init), 
         5: create_model_5(init), 
         6: create_model_6(init),
         7: create_model_0(init),
         8: create_model_3(init)}

included_model = [0, 1, 2, 3, 4, 6, 7, 8]

for i in included_model:
    model[i].compile(loss='binary_crossentropy', optimizer=Adam(1e-03), metrics=['acc',f1])
    print('\nModel ' + str(i))
    model[i].summary()


os.environ["THEANO_FLAGS"]='device=gpu0'


# In[19]:


paths, labels = getTrainDataset()

# calculate_weight
num_label = np.sum(labels, axis=0)
print(num_label)
log_num_label = np.log(num_label)
class_weight = {}
for i in range(len(log_num_label)):
    class_weight[i] = min(log_num_label) / log_num_label[i]


# In[20]:


def switch_data(pathsT, labelsT, pathsV, labelsV, indexT, target_label):
    
    last_row_without_target_label = len(labelsV)
    while last_row_without_target_label > 0:
        last_row_without_target_label -= 1
        if labelsV[last_row_without_target_label, target_label] != 1.0:
            break
    
    pathT_temp = np.array(pathsT[indexT]).reshape(1,)
    labelT_temp = np.array(labelsT[indexT, :]).reshape(1,28)
    pathV_temp = np.array(pathsV[last_row_without_target_label]).reshape(1,)
    labelV_temp = np.array(labelsV[last_row_without_target_label, :]).reshape(1,28)
    
    pathsT = np.delete(pathsT, indexT, axis=0)
    labelsT = np.delete(labelsT, indexT, axis=0)
    pathsV = np.delete(pathsV, last_row_without_target_label, axis=0)
    labelsV = np.delete(labelsV, last_row_without_target_label, axis=0)
    
    pathsT = np.concatenate((pathsT, pathV_temp), axis=0)
    labelsT = np.concatenate((labelsT, labelV_temp), axis=0)
    pathsV = np.concatenate((pathsV, pathT_temp), axis=0)
    labelsV = np.concatenate((labelsV, labelT_temp), axis=0)
    
    return pathsT, labelsT, pathsV, labelsV


# In[21]:


def search_first_label(labelsT, label_index, begin_index):
    while begin_index < len(labelsT):
        if labelsT[begin_index, label_index] != 0.0:
            break
        else:
            begin_index += 1
    return begin_index


# In[22]:


# divide to 
keys = np.arange(paths.shape[0], dtype=np.int)  
np.random.seed(SEED)
np.random.shuffle(keys)
lastTrainIndex = int((1-VAL_RATIO) * paths.shape[0])

pathsTrain = paths[0:lastTrainIndex]
labelsTrain = labels[0:lastTrainIndex]
pathsVal = paths[lastTrainIndex:]
labelsVal = labels[lastTrainIndex:]

#rint('Before balancing label_27:')
#print(paths.shape, labels.shape)
#print(pathsTrain.shape, labelsTrain.shape, pathsVal.shape, labelsVal.shape)


# In[23]:


num_label_Train = np.sum(labelsTrain, axis=0)
target_num_label_train = int(num_label[27] * (1 - VAL_RATIO))

#print('train -> val')
while num_label_Train[27] > target_num_label_train:
    begin_index = search_first_label(labelsTrain, 27, 0)
    pathsTrain, labelsTrain, pathsVal, labelsVal =         switch_data(pathsTrain, labelsTrain, pathsVal, labelsVal, begin_index, 27)
    num_label_Train = np.sum(labelsTrain, axis=0)

#print('val -> train')
while num_label_Train[27] < target_num_label_train:
    begin_index = search_first_label(labelsVal, 27, 0)
    pathsVal, labelsVal, pathsTrain, labelsTrain  =         switch_data(pathsVal, labelsVal, pathsTrain, labelsTrain, begin_index, 27)
    num_label_Train = np.sum(labelsTrain, axis=0)

#print('After balancing label_27:')
#print(paths.shape, labels.shape)
#print(pathsTrain.shape, labelsTrain.shape, pathsVal.shape, labelsVal.shape)


# In[24]:


tg = ProteinDataGenerator(pathsTrain, labelsTrain, BATCH_SIZE, SHAPE, use_cache=True, augment = True, shuffle = False)
vg = ProteinDataGenerator(pathsVal, labelsVal, BATCH_SIZE, SHAPE, use_cache=True, shuffle = False)

earlystopping = EarlyStopping(monitor='val_f1', patience=10, verbose=1, mode='max')
# https://keras.io/callbacks/#modelcheckpoint
reduceLROnPlato = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='min')


epochs = 40

use_multiprocessing = False # DO NOT COMBINE MULTIPROCESSING WITH CACHE! 
workers = 1 # DO NOT COMBINE MULTIPROCESSING WITH CACHE! 

for i in included_model:
    print('\nModel ' + str(i))
    best_file = Path('../model/best' + str(i) + '.model')
    if best_file.is_file():
        print('\tbest' + str(i) + '.model exists.')
    else:
        checkpoint = ModelCheckpoint('../model/best' + str(i) + '.model', monitor='val_f1', verbose=1,                                      save_best_only=True, save_weights_only=False, mode='max', period=1)
        hist = model[i].fit_generator(
            tg,
            steps_per_epoch=len(tg),
            validation_data=vg,
            validation_steps=8,
            epochs=epochs,
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            class_weight=class_weight,
            verbose=1,
            callbacks=[checkpoint,earlystopping]
        )
        draw_process(hist, i, 'phase_1')


# In[27]:


for i in included_model:
    print('\nModel ' + str(i))
    model_file = Path('../model/tuned' + str(i) + '.model')
    if model_file.is_file():
        print('\ttuned' + str(i) + '.model exists.')
    else:
        # fine-tuning for DNN

        model[i] = load_model('../model/model/best' + str(i) + '.model', custom_objects={'f1': f1}) #, 'f1_loss': f1_loss})
        checkpoint = ModelCheckpoint('../model/tuned' + str(i) + '.model', monitor='val_f1', verbose=1,                                          save_best_only=True, save_weights_only=False, mode='max', period=1)

        for layer in model[i].layers:
            layer.trainable = False

        for L in range(-1, -8, -1):
            model[i].layers[L].trainable = True

        model[i].compile(loss=f1_loss,
                         optimizer=Adam(lr=1e-4),
                         metrics=['accuracy', f1])

        hist = model[i].fit_generator(
            tg,
            steps_per_epoch=len(tg),
            validation_data=vg,
            validation_steps=8,
            epochs=10,
            class_weight=class_weight,
            use_multiprocessing=use_multiprocessing, # you have to train the model on GPU in order to this to be benefitial
            workers=workers, # you have to train the model on GPU in order to this to be benefitial
            verbose=1,
            max_queue_size=4,
            callbacks=[checkpoint,earlystopping]
        )
        draw_process(hist, i, 'phase_2')


# In[28]:


for i in included_model:
    print('\nmodel ' + str(i))
    model_file = Path('../model/tuned_again' + str(i) + '.model')
    if model_file.is_file():
        print('\ttuned_again' + str(i) + '.model exists.')
    else:
        # fine-tuning for CNN

        model[i] = load_model('../model/tuned' + str(i) + '.model', custom_objects={'f1': f1, 'f1_loss': f1_loss})
        checkpoint = ModelCheckpoint('../model/tuned_again' + str(i) + '.model', monitor='val_f1', verbose=1,                                          save_best_only=True, save_weights_only=False, mode='max', period=1)

        for layer in model[i].layers:
            layer.trainable = True
        
        for L in range(-1, -8, -1):
            model[i].layers[L].trainable = False

        model[i].compile(loss=f1_loss,
                         optimizer=Adam(lr=1e-4),
                         metrics=['accuracy', f1])

        hist = model[i].fit_generator(
            tg,
            steps_per_epoch=len(tg),
            validation_data=vg,
            validation_steps=8,
            epochs=5,
            class_weight=class_weight,
            use_multiprocessing=use_multiprocessing, # you have to train the model on GPU in order to this to be benefitial
            workers=workers, # you have to train the model on GPU in order to this to be benefitial
            verbose=1,
            max_queue_size=4,
            callbacks=[checkpoint,earlystopping]
        )
        draw_process(hist, i, 'phase_3')


# In[29]:


for i in included_model:
    print('\nmodel ' + str(i))
    model_file = Path('../model/tuned_again_again' + str(i) + '.model')
    if model_file.is_file():
        model[i] = load_model('../model/tuned_again_again' + str(i) + '.model', custom_objects={'f1': f1, 'f1_loss': f1_loss})
        print('\ttuned_again_again' + str(i) + '.model loaded.')
    else:
        # fine-tuning for DNN

        model[i] = load_model('../model/tuned_again' + str(i) + '.model', custom_objects={'f1': f1, 'f1_loss': f1_loss})
        checkpoint = ModelCheckpoint('../model/tuned_again_again' + str(i) + '.model', monitor='val_f1', verbose=1,                                          save_best_only=True, save_weights_only=False, mode='max', period=1)

        for layer in model[i].layers:
            layer.trainable = False

        for L in range(-1, -8, -1):
            model[i].layers[L].trainable = True

        model[i].compile(loss=f1_loss,
                         optimizer=Adam(lr=1e-4),
                         metrics=['accuracy', f1])

        hist = model[i].fit_generator(
            tg,
            steps_per_epoch=len(tg),
            validation_data=vg,
            validation_steps=8,
            epochs=3,
            class_weight=class_weight,
            use_multiprocessing=use_multiprocessing, # you have to train the model on GPU in order to this to be benefitial
            workers=workers, # you have to train the model on GPU in order to this to be benefitial
            verbose=1,
            max_queue_size=4,
            callbacks=[checkpoint,earlystopping]
        )
        draw_process(hist, i, 'phase_4')

print('End of training.')
