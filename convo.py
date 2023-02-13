from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential

# Read the dataset "pythonProject\dataset" and create a image generator with augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

training_set = train_datagen.flow_from_directory('dataset', target_size=(28, 28), batch_size=64, class_mode='binary')

model = Sequential([
    Conv2D(128, kernel_size=(5, 5), strides=1, padding='same', activation='relu', input_shape=(28, 28, 3)),
    MaxPool2D(pool_size=(3, 3), strides=2, padding='same'),
    Conv2D(64, kernel_size=(2, 2), strides=1, activation='relu', padding='same'),
    MaxPool2D((2, 2), 2, padding='same'),
    Conv2D(32, kernel_size=(2, 2), strides=1, activation='relu', padding='same'),
    MaxPool2D((2, 2), 2, padding='same'),

    Flatten(),
    Dense(units=512, activation='relu'),
    Dropout(rate=0.25),
    Dense(29, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()
model.fit(training_set, epochs=35)

# Save the model
model.save('model.h5')
