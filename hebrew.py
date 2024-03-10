import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Dense, Flatten
import matplotlib.pyplot as plt


meta = {'image_size':(32,32), 'batch_size':32}


train_ds = keras.utils.image_dataset_from_directory(
  '..\\hebrew_project-master\\hebrew_letters_dataset',
  validation_split=0.2,
  subset="training",
  label_mode='categorical',
  seed=123,
  image_size = meta['image_size'],
  batch_size = meta['batch_size'])



val_ds = keras.utils.image_dataset_from_directory(
  '..\\hebrew_project-master\\hebrew_letters_dataset',
  validation_split=0.2,
  subset="validation",
  label_mode='categorical',
  seed=123,
  image_size = meta['image_size'],
  batch_size = meta['batch_size'])



epochs = 30
class_names = train_ds.class_names



try:
  model = keras.models.load_model('model')
except:
  model = keras.Sequential()
  #Layer 1
  model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = (32, 32, 3)))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Dropout(0.25))

  #Layer 2
  model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', padding='same'))
  model.add(Dropout(0.25))

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
  model.add(Dense(18, activation='softmax'))
  #compile the CNN

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


data_of = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=2,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)

saved_file = keras.callbacks.ModelCheckpoint(
    'model.keras',
    monitor="val_accuracy",
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
    initial_value_threshold=None,
)


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











