# import required libraries
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam

# set seed for reproducibility
tf.random.set_seed(42)

# Image generator with augmentation
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   rotation_range=20, 
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2,
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True)

# create train set generator
training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size=(224, 224),
                                                 batch_size=64, 
                                                 class_mode='categorical')

# create validation set generator
val_datagen = ImageDataGenerator(rescale=1./255)

validation_set = val_datagen.flow_from_directory('dataset/validation', 
                                                  target_size=(224, 224),
                                                  batch_size=64, 
                                                  class_mode='categorical')

URL = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(224, 224, 3))

feature_extractor.trainable = False

model = Sequential([
    feature_extractor,
    layers.Dense(29)
])


# compile the model
model.compile(optimizer=Adam(lr=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# print model summary
model.summary()

# train the model
history = model.fit(training_set, 
                    epochs=10,
                    validation_data=validation_set)

# save the model
model.save('modeltole.h5')