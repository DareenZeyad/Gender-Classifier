# Step 4 - Creating and training Model
# Training Imports
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import random
import cv2
import os
import glob

# Global Variables
lerningRate = 1e-3
numEpochs = random.randint(100, 200)
currOptimizer = Adam(lr = lerningRate, decay = lerningRate/numEpochs)
datasetLocation = "newDataset/**/*"
image_dimensions = (96, 96, 3) # (height, width, depth)
numBatches = 32
Xmatrix = []
Yvector = []

# Part 1 - Filling the dataset with the files
# Adding Paths to Filelist
filesList = [filename for filename in glob.glob(datasetLocation, recursive = True) if not os.path.isdir(filename)]
random.seed(random.randint(40, 80))
random.shuffle(filesList)

# Resizing Each Photo and appending it to dataset
cntr = 0
for filename in filesList:
    image = cv2.imread(filename)
    image = cv2.resize(image, (image_dimensions[0], image_dimensions[1]))
    image = img_to_array(image)
    image = np.array(image, dtype = "float") / 255.0
    Xmatrix.append(image)
    currlabel = filename.split(os.path.sep)[-2]
    currlabel = 0 if currlabel == "man" else 1
    Yvector.append([currlabel])
    cntr += 1
    print(cntr)

# Part 2 - Data Pre-processing
Xmatrix = np.array(Xmatrix, dtype = "float")
Yvector = np.array(Yvector)
(X_train, X_test, Y_train, Y_test) = train_test_split(Xmatrix, Yvector, test_size = 0.2,random_state = random.randint(40, 80))
Y_train = to_categorical(Y_train, num_classes = 2)
Y_test = to_categorical(Y_test, num_classes = 2)

# Part 3 - Dataset Augmentation
data_augemntation = ImageDataGenerator(data_format = "channels_first", rescale = 1./255, rotation_range = 25, width_shift_range = 0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, vertical_flip = False,
                         horizontal_flip = True, fill_mode="nearest")

# Part 4 - Creating Model
# Model Imports
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense

# Model Creation
# Initializing Model
model = Sequential()

# Layer 1 - Convolution-Normalizing-MaxPooling - Dropout 0.25
model.add(Conv2D(32, (3, 3), padding = "same", input_shape = (96, 96, 3), activation = 'relu'))
model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D(pool_size = (3, 3)))
model.add(Dropout(0.25))

# Layer 2 - Convolution-Convolution-Normalizing-MaxPooling - Dropout 0.25
model.add(Conv2D(64, (3, 3), padding = "same", activation = 'relu'))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(64, (3, 3), padding = "same", activation = 'relu'))
model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

# Layer 3 - Convolution-Convolution-Normalizing-MaxPooling - Dropout 0.25
model.add(Conv2D(128, (3, 3), padding = "same", activation = 'relu'))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(128, (3, 3), padding = "same", activation = 'relu'))
model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

# Layer 4 - Convolution-Convolution-Normalizing-MaxPooling - Dropout 0.5
model.add(Conv2D(256, (3, 3), padding = "valid", activation = 'relu'))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(256, (3, 3), padding = "valid", activation = 'relu'))
model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.5))

# Layer 5 - Flattening
model.add(Flatten())

# Layer 6 - FCC Input Layer - Dropout 0.5
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Layer 7 - FCC Output Layer - Activation Softmax
model.add(Dense(2))
model.add(Activation("softmax"))

# Compiling
model.compile(optimizer = currOptimizer, loss = "binary_crossentropy", metrics = ["accuracy"])

# Model Training
trainedModel = model.fit_generator(data_augemntation.flow(X_train, Y_train, batch_size = numBatches),
                        validation_data = (X_test, Y_test),
                        steps_per_epoch = len(X_train) // numBatches,
                        epochs = numEpochs, verbose = 1)

# Saving Model - Weights
model.save('PreTrainedModel.h5')
# model.save_weights('savedModel.h5')