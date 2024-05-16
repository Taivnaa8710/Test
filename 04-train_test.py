import os
import shutil
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from PIL import ImageFile
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from efficientnet.tfkeras import EfficientNetB7 #EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

import warnings
warnings.simplefilter('error', Image.DecompressionBombWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000

devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(devices[0], True)
    print("Success in setting memory growth")
except:
    print("Failed to set memory growth, invalid device or cannot modify virtual devices once initialized.")

batch_size = 32
epoch = 30
num_of_channels = 3
fold_num = 1
dataset_folder_name = '/Dataset/split_dataset/'
source_files = []
X=[]
Y=[]
img_rows, img_cols = 380, 380
train_path = os.path.join(dataset_folder_name, 'train')
validation_path = os.path.join(dataset_folder_name, 'val')
test_path = os.path.join(dataset_folder_name, 'test')

class_labels = ['fake', 'real']
number_of_class_labels = len(class_labels)

def get_model():
    
    efficient_net = EfficientNetB7(
        weights = 'imagenet',
        input_shape = (img_rows, img_cols, num_of_channels),
        include_top = False,
        pooling = 'max'
    )
    
    model = Sequential()
    model.add(efficient_net)
    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dense(units = 1, activation = 'sigmoid'))
    model.summary()
    
    model.compile(optimizer = Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model
model = get_model()
        

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.2,
    zoom_range=0.1,
    horizontal_flip = True,
    fill_mode="nearest")


    # Start ImageClassification Model
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_rows, img_cols),
    color_mode = "rgb",
    batch_size=batch_size,
    class_mode="binary",
    shuffle = True,
    subset='training')
validation_datagen = ImageDataGenerator(rescale=1./255)
    
validation_generator = validation_datagen.flow_from_directory(
    validation_path,
    target_size=(img_rows, img_cols),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="binary",  # only data, no labels
    shuffle=False)
checkpoint_filepath = '/tmp_checkpoint'
print('Creating Directory: ' + checkpoint_filepath)
os.makedirs(checkpoint_filepath, exist_ok=True)
    
custom_callbacks = [
    EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        patience = 5,
        verbose = 1
    ),
    ModelCheckpoint(
        filepath = os.path.join(checkpoint_filepath, '380x380_best_model.h5'),
        monitor = 'val_loss',
        mode = 'min',
        verbose = 1,
        save_best_only = True
    )
]

    # fit model
num_epochs = epoch
history = model.fit(train_generator,
                    epochs=num_epochs,
                    steps_per_epoch = len(train_generator),
                    validation_data = validation_generator,
                    validation_steps = len(validation_generator),
                    callbacks = custom_callbacks)
    
predictions = model.predict(validation_generator, verbose=1)
y_predictions = np.argmax(predictions, axis=1)
true_classes = validation_generator.classes
    
    # evaluate validation performance
    #print("***Performance on Validation data***")
    #val_acc, val_prec, val_f1Score = my_metrics(true_classes, y_predictions)
    # Save the training history to a CSV file
print(history.history)
history_df = pd.DataFrame(history.history)
history_df.to_csv('/results/B7_batch_32_fold_{}_history.csv'.format(fold_num), index=False)
    # Visualize the training history for the current fold
plt.figure(figsize=(12, 6))
    
plt.subplot(1, 2, 1)
plt.plot(history_df['loss'], label='Training Loss')
plt.plot(history_df['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.title('Training and Validation Loss')
plt.legend()
    
plt.subplot(1, 2, 2)
plt.plot(history_df['accuracy'], label='Training Accuracy')
plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.savefig('/results/B7_batch_32_fold_{}_history_plot.png'.format(fold_num))  # Save the plot as an image
    #plt.show()
print("==============TEST RESULTS============")
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    classes=['real','fake'],
    target_size=(img_rows, img_cols),
    batch_size=1,
    color_mode="rgb",
    class_mode=None,
    shuffle=False 
)
# load the saved model that is considered the best
best_model = load_model(os.path.join(checkpoint_filepath, '380x380_best_model.h5'))
test_generator.reset()
predictions = best_model.predict(test_generator, verbose=1)
    
    # Your data splitting logic
    
    # ... (previous code for data preparation)

    # Training and evaluation

    # ... (previous code for model training and evaluation)

    # Save predictions for this fold to a CSV file
test_results = pd.DataFrame({
    "Filename": test_generator.filenames,
    "Prediction": predictions.flatten()
})
print(test_results)
csv_filename = '/results/380x380_B7_batch_32_fold_{}_predictions.csv'.format(fold_num)
test_results.to_csv(csv_filename, index=False)
print("Predictions for fold {} saved to {}".format(fold_num, csv_filename))
    
y_predictions = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
    
   