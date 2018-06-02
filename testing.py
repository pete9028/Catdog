# this code can't run together with jupyter
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import tensorflow as tf

# dimensions of our images.
img_width, img_height = 150, 150

input_shape = (img_width, img_height, 3)
test_model = Sequential()

test_model.add(Conv2D(32, (3, 3), input_shape=input_shape))
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(2, 2)))

test_model.add(Conv2D(32, (3, 3)))
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(2, 2)))

test_model.add(Conv2D(64, (3, 3)))
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(2, 2)))

test_model.add(Flatten())
test_model.add(Dense(64))
test_model.add(Activation('relu'))
test_model.add(Dropout(0.5))
test_model.add(Dense(1))
test_model.add(Activation('sigmoid'))

test_model = load_model('first_model.h5')
def predict(basedir, model):
    for i in range(1,11):
        path = basedir + str(i) + '.jpg'
    
        img = load_img(path,False,target_size=(img_width,img_height))
        x = img_to_array(img)
        x = x/255
        x = np.expand_dims(x, axis=0)
        preds = model.predict_classes(x)
        probs = model.predict_proba(x)
        if (preds[0][0]==0):
            probs[0][0]=1-probs[0][0]
        print("Picture",i,":  ",preds[0][0],"     ",round(probs[0][0],3))

def predictclass(basedir, model):
    c=""
    for i in range(1,11):
        path = basedir + str(i) + '.jpg'
    
        img = load_img(path,False,target_size=(img_width,img_height))
        x = img_to_array(img)
        x = x/255
        x = np.expand_dims(x, axis=0)
        preds = model.predict_classes(x)
        probs = model.predict_proba(x)
        if (preds[0][0]==0):
            probs[0][0]=1-probs[0][0]
        if (preds[0][0]==1):
            c="dog"
        else:
            c="cat"
        print("Picture",i,": ",c,"    ",round(probs[0][0],3))

print("Testing model")

print()
print("20 Pictures from training set")    
print("cats:")
basedir = "data/train/cats/cat."
print("            Class : Probability")
predict(basedir, test_model)
print()
print("dogs:")
basedir = "data/train/dogs/dog."
print("            Class : Probability")
predict(basedir, test_model)
print()
print("0 - cat")
print("1 - dog")
print()
print("20 Pictures from testing set")
print("Predict 10 pictures of cat:")
basedir = "test1/cat/"
print("            Class : Probability")
predictclass(basedir, test_model)
print()
print("Predict 10 pictures of dog:")
basedir = "test1/dog/"
print("            Class : Probability")
predictclass(basedir, test_model)

