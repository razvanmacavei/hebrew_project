
from tensorflow import keras
from tensorflow.keras import layers, datasets
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Dense, Flatten
from keras.utils import np_utils
<<<<<<< HEAD
import tensorflow as tf
import random
import matplotlib.pyplot as plt
=======


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
>>>>>>> f555c8dc386cc30d70b80567180778f552270e83


meta = {'image_size':(32,32), 'batch_size':32}


train_ds = tf.keras.utils.image_dataset_from_directory(
  'D:\AAASubjects\proiect\hebrew_letters',
  validation_split=0.2,
  subset="training",
  label_mode='categorical',
  seed=123,
  image_size = meta['image_size'],
  batch_size = meta['batch_size'])



val_ds = tf.keras.utils.image_dataset_from_directory(
  'D:\AAASubjects\proiect\hebrew_letters',
  validation_split=0.2,
  subset="validation",
  label_mode='categorical',
  seed=123,
  image_size = meta['image_size'],
  batch_size = meta['batch_size'])



epochs = 30
class_names = train_ds.class_names



try:
  model = tf.keras.models.load_model('model')
except:
  model = keras.Sequential()
  #Layer 1
  model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = (32, 32, 3)))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Dropout(0.25))

  #Layer 2
  # model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', padding='same'))
  # # model.add(MaxPooling2D(pool_size = (2,2)))
  # model.add(Dropout(0.25))
  
  #Layer 2'
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


data_of = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=2,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)

saved_file = tf.keras.callbacks.ModelCheckpoint(
    'model',
    monitor="val_accuracy",
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
    options=None,
    initial_value_threshold=None,
)

# model.summary()
output = model.fit(train_ds, batch_size = meta['batch_size'], epochs = epochs, validation_data = val_ds, 
                   callbacks = [saved_file, data_of])


acc = output.history['accuracy']
val_acc = output.history['val_accuracy']

loss = output.history['loss']
val_loss = output.history['val_loss']

epochs_range = range(len(output.history['loss']))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
    







