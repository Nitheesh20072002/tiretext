import numpy as np
import  os
import cv2
from  sklearn.model_selection  import train_test_split
import  matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import  Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D

import  pickle
###################################################

path='D:/pycharmprojects/tiretext/myData'
picklepath='D:/pycharmprojects/tiretext/A'
testRatio=0.2
valRation=0.2
imageDimension=(32,32,3)

batchSizeVal= 50
epochsVal = 1
stepsPerEpoch = 2000

######################################################
images = []
classNo=[]
myList= os.listdir(path)
print("Total No of classes detected : ",len(myList))
noOfClases=len(myList)
print("Importing classes......")
for x in range(0,noOfClases):
    myPicList=os.listdir(path+"/"+str(x))
    for y in myPicList:
        curImg=cv2.imread(path+"/"+str(x)+"/"+y)
        curImg=cv2.resize(curImg,(imageDimension[0],imageDimension[1]))
        images.append(curImg)
        classNo.append(x)
    print(x,end=" ")
print(" ")

images=np.array(images)
classNo=np.array(classNo)

# print(images.shape)
# print(classNo.shape)

######## Splitting data
X_train,X_test,y_train,y_test=train_test_split(images,classNo,test_size=testRatio)
# print(X_train.shape)
X_train,X_validation,y_train,y_validation=train_test_split(X_train,y_train,test_size=valRatio)

numOfSamples=[]
for x in range(0,noOfClases):
    numOfSamples.append(len(np.where(y_train==x)[0]))
print(numOfSamples)

plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClases),numOfSamples)
plt.title("NO of Images for each class")
plt.xlabel("Class Id")
plt.ylabel("Number of Images")
plt.show()

def Preprocessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255
    return  img

##Preprocessing
X_train=np.array(list(map(Preprocessing,X_train)))
X_test=np.array(list(map(Preprocessing,X_test)))
X_validation=np.array(list(map(Preprocessing,X_validation)))

##depth
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)

dataGen=ImageDataGenerator(width_shift_range=0.1,
                           height_shift_range=0.1,
                           zoom_range=0.2,
                           shear_range=0.1,
                           rotation_range=10)
dataGen.fit(X_train)

######one hot encode(not necessary for neural network)
y_train=to_categorical(y_train,noOfClases)
y_test=to_categorical(y_test,noOfClases)
y_validation=to_categorical(y_validation,noOfClases)

###model
def myModel():
    noOffilters=60
    sizeOfFilter1=(5,5)
    sizeOfFilter2=(3,3)
    sizeOfPool=(2,2)
    noOfNode=500

    model=Sequential()
    model.add((Conv2D(noOffilters,sizeOfFilter1,input_shape=(imageDimension[0],imageDimension[1],1),activation='relu')))
    model.add((Conv2D(noOffilters,sizeOfFilter1,activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOffilters//2,sizeOfFilter2,activation='relu')))
    model.add((Conv2D(noOffilters//2,sizeOfFilter2,activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flaten())
    model.add(Dense(noOfNode,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfNode, activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracye'])
    return  model;

model=myModel()
print(model.summary())



history=model.fit_generator(dataGen.flow(X_train,y_train),batch_size=batchSizeVal,steps_per_epoch=stepsPerEpoch,
                    epochs=epochsVal, validation_data=(X_validation,y_validation),shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
scor= model.evaluate(X_test,y_test,verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy = ',score[1])


pickle_out=open(picklepath+"/"+"model_trained.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()