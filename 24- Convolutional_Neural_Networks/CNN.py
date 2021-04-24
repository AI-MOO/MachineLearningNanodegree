# Convolutional Neural Network. 

# Installing the required libraries
# Theano, Tensorflow, Keras 
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Conv2D   
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# Sequential 
classifier = Sequential()

# Step1: Convolution 
# n_filters(feature maps) = 32, kernel_size: rows = 3, cols = 3 
# input image shape 64x64 RGB 
classifier.add(Conv2D(32, kernel_size = (3,3), input_shape = (64, 64, 3), activation = 'relu')) 

# Step2: Pooling
classifier.add(MaxPooling2D(pool_size = (2,2))) 

# Adding a second convolutional layer
classifier.add(Conv2D(32, kernel_size = (3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2))) 

# Step3: Flatten 
classifier.add(Flatten())

# Step4: Full Connection 
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compile the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Part 2 Fitting the CNN to the images 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
classifier.fit(
        training_set,
        steps_per_epoch=250,    # no.unique images in training set / batch size
        epochs=35,
        validation_data= test_set,
        validation_steps=2000)




