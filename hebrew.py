
from PIL import Image
import glob
from tensorflow import keras
from tensorflow.keras import layers, datasets
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Dense, Flatten
import matplotlib.pyplot as plt
import random
from keras.utils import np_utils


meta = {'image_size':(64,64), 'batch_size':32}

    
(X_train, y_train), (X_test, y_test) = keras.utils.image_dataset_from_directory(
    directory='D:\AAASubjects\Proiect internship - evreiesti\letters',
    labels='inferred',
    label_mode='categorical',
    batch_size=meta['batch_size'],
    image_size=meta['image_size'])

X_train = X_train.astype('float')
X_test = X_test.astype('float')

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


epochs = 1
class_names = ['alef', 'bet', 'gimel', 'dalet', 'hey', 'waw', 'zain', 'het', 'tet', 'yod']


model = keras.Sequential()
#Layer 1
model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = (32, 32, 3)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

#Layer 2
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

#Layer 3
model.add(Conv2D(128, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())  
model.add(Dense(512, activation = 'relu', kernel_initializer = 'uniform'))
model.add(Dropout(0.30))
model.add(Dense(22, activation = 'softmax'))
#compile the CNN

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
output = model.fit(X_train, y_train, batch_size = meta['batch_size'], epochs = epochs, validation_data = (X_test, y_test))

y_pred=model.predict(X_test)
index= random.randint(0,9999)
plt.imshow(X_test[index])

y_pred=model.predict(X_test)
index= random.randint(0,9999)
plt.imshow(X_test[index])

print(class_names[y_pred[index].argmax()] + "\n")
print(class_names[int(y_test[index].argmax())]+ "\n")
if(class_names[y_pred[index].argmax()]==class_names[int(y_test[index].argmax())]):
    print("the predicition of the image was correct")
else: print("improve the model")




