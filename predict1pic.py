# this code can't run together with jupyter
# To predict picture, you need to place picture in the folder "testing"
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
def predictclass(flies, model):
    c=""
    path = basedir + flies

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
    print("This picture has a",c)
    #uncomment line below to see probability
    #print("The probability is",round(probs[0][0],3))
basedir = "testing/"
print("Predict picture")

# To predict picture, you need to place picture in the folder "testing"
#picture file name that you want to predict eg. dog3.jpg
files = "example.jpg"
predictclass(files, test_model)

