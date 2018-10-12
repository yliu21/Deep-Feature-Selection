# Train models on each time points
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import Input, Dense , Dropout , Flatten, BatchNormalization
from keras.layers import MaxPooling2D, AveragePooling2D 
from keras.layers.convolutional import Conv2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold 
plt.switch_backend('Agg')

seed=1234
np.random.seed(seed)
os.chdir('/work/04940/yliu21/maverick/ADNI_DTI')
dim1=91
dim2=109
time_point='6m'

# Load data
# Extract slices
def extract_slice(X,y,dim1=dim1,dim2=dim2):
   n=X.shape[0]
   temp =np.moveaxis(X[:,:,:,20:70],3,1)
   temp =temp.reshape((n*50,dim1,dim2))
   temp =temp.reshape((-1,dim1, dim2, 1))
   img = np.swapaxes(temp,1,2)
   label = np.repeat(y,50,0)
   print(img.shape)
   print(label.shape)
   return([img,label])

def data_gen(time_point=time_point):
   img=np.load('./data/img_adnc_'+time_point+'.npy')
   label=to_categorical(np.load('./data/label_adnc_'+time_point+'.npy'))
   img_train_val, img_test, label_train_val, label_test = train_test_split(img, label, test_size=0.2, random_state=seed, 
   shuffle=True)
   img_train, img_val, label_train, label_val = train_test_split(img_train_val, label_train_val, test_size=0.2, random_state=seed, 
   shuffle=True)
   X_train, y_train=extract_slice(img_train, label_train)
   X_train, y_train=shuffle(X_train, y_train)
   X_val, y_val=extract_slice(img_val,label_val)
   X_val, y_val=shuffle(X_val, y_val)
   X_test, y_test=extract_slice(img_test,label_test)
   X_test, y_test=shuffle(X_test, y_test)
   return([X_train,y_train, X_val, y_val, X_test, y_test])

X_train,y_train, X_val, y_val, X_test, y_test = data_gen(time_point=time_point)

# Construct the model structure
input=Input(shape=(109,91,1), name='input_images')

conv11=Conv2D(64, (3,3), padding='same', activation='relu', name='conv11', trainable=True)(input)
conv12=Conv2D(64, (3,3), padding='same', activation='relu', name='conv12', trainable=True)(conv11)
batchnorm1=BatchNormalization()(conv12)
dropout1=Dropout(rate=0.4)(batchnorm1)
pool1=MaxPooling2D(2,name='pool1')(dropout1)

conv21=Conv2D(128, (3,3), padding='same', activation='relu', name='conv21', trainable=True)(pool1)
conv22=Conv2D(128, (3,3), padding='same', activation='relu', name='conv22', trainable=True)(conv21)
batchnorm2=BatchNormalization()(conv22)
dropout2=Dropout(rate=0.4)(batchnorm2)
pool2=MaxPooling2D(2,name='pool2')(dropout2)

conv31=Conv2D(256, (3,3), padding='same', activation='relu', name='conv31', trainable=True)(pool2)
conv32=Conv2D(256, (3,3), padding='same', activation='relu', name='conv32', trainable=True)(conv31)
conv33=Conv2D(256, (3,3), padding='same', activation='relu', name='conv33', trainable=True)(conv32)
conv34=Conv2D(256, (3,3), padding='same', activation='relu', name='conv34', trainable=True)(conv33)
batchnorm3=BatchNormalization()(conv34)
dropout3=Dropout(rate=0.4)(batchnorm3)
pool3=MaxPooling2D(2, name='pool3')(dropout3)

# conv41=Conv2D(512, (3,3), padding='same', activation='relu', name='conv41', trainable=True)(pool3)
# conv42=Conv2D(512, (3,3), padding='same', activation='relu', name='conv42', trainable=True)(conv41)
# conv43=Conv2D(512, (3,3), padding='same', activation='relu', name='conv43', trainable=True)(conv42)
# conv44=Conv2D(512, (3,3), padding='same', activation='relu', name='conv44', trainable=True)(conv43)
# batchnorm4=BatchNormalization()(conv44)
# dropout4=Dropout(rate=0.4)(batchnorm4)
# pool4=MaxPooling2D(2, name='pool4')(dropout4)

# conv51=Conv2D(512, (3,3), padding='same', activation='relu', name='conv51', trainable=True)(pool4)
# conv52=Conv2D(512, (3,3), padding='same', activation='relu', name='conv52', trainable=True)(conv51)
# conv53=Conv2D(512, (3,3), padding='same', activation='relu', name='conv53', trainable=True)(conv52)
# conv54=Conv2D(512, (3,3), padding='same', activation='relu', name='conv54', trainable=True)(conv53)
# batchnorm5=BatchNormalization()(conv54)
# dropout5=Dropout(rate=0.4)(batchnorm5)
# pool5=MaxPooling2D(2, name='pool5')(dropout5)

# gap=AveragePooling2D((3,2),name='gap')(pool5)
# flat=Flatten(name='flat')(gap)
flat=Flatten(name='flat')(pool3)
# dense61=Dense(4096, activation='relu', name='dense61')(flat)
# # drop61=Dropout(rate=0.4)(dense61)
# dense62=Dense(4096, activation='relu', name='dense62')(dense61)
# drop62=Dropout(rate=0.4)(dense62)
dense63=Dense(1024, activation='relu', name='dense63')(flat)
# drop63=Dropout(rate=0.4)(dense63)
pred=Dense(2, activation='softmax', name='prob')(dense63)

model_classify=Model(input=input, output=pred)

# Compile the model
adam= Adam(lr=0.000005)
sgd=SGD(lr=0.00001)
model_classify.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
# Callbacks
checkpoint=ModelCheckpoint('./results/models/longitudinal_axial/random_by_sub/vgg_'+time_point+'.hdf5', monitor='val_acc',
                          save_best_only=True, save_weights_only=True, verbose=1)
earlystopping=EarlyStopping(monitor='val_acc', min_delta=0.00001, patience=15, verbose=1, mode='auto')

# Fit the model
# first 100 epochs no early stopping
# model_classify.load_weights('./results/models/longitudinal_axial/vgg_'+time_point+'.hdf5')

history=model_classify.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), batch_size=64, callbacks=[checkpoint],verbose=2)

# Load best model
model_classify.load_weights('./results/models/longitudinal_axial/random_by_sub/vgg_'+time_point+'.hdf5')

# Accuracy on test dataset
loss_test, acc_test=model_classify.evaluate(X_test, y_test)
print('Test Accuracy = %.4f' %acc_test)

# Plot the training history
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy (Test Acc = %.4f)' %acc_test)
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train','validation'], loc='upper left')
plt.savefig('./results/models/longitudinal_axial/random_by_sub/'+time_point+'_acc.png')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train','validation'], loc='upper left')
plt.savefig('./results/models/longitudinal_axial/random_by_sub/'+time_point+'_loss.png')

