import numpy as np                   # advanced math library
import matplotlib.pyplot as plt      # MATLAB like plotting routines
import random                        # for generating random numbers

from keras.datasets import mnist     # MNIST dataset is included in Keras
from keras.models import Sequential  # Model type to be used

from keras.layers.core import Dense, Dropout, Activation # Types of layers to be used in our model
from keras.utils import np_utils                         # NumPy related tools
# import some additional tools

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Flatten
from keras.layers.normalization import BatchNormalization


# just a little function for pretty printing a matrix
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

if __name__ == 'main':

    # The MNIST data is split between 60,000 28 x 28 pixel training images and 10,000 28 x 28 pixel images
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print("X_train shape", X_train.shape)
    print("y_train shape", y_train.shape)
    print("X_test shape", X_test.shape)
    print("y_test shape", y_test.shape)

    plt.rcParams['figure.figsize'] = (9,9) # Make the figures a bit bigger

    for i in range(9):
        plt.subplot(3,3,i+1)
        num = random.randint(0, len(X_train))
        plt.imshow(X_train[num], cmap='gray', interpolation='none')
        plt.title("Class {}".format(y_train[num]))
        
    plt.tight_layout()

    # now print!        
    matprint(X_train[num])

    X_train = X_train.reshape(60000, 28, 28, 1) #add an additional dimension to represent the single-channel
    X_test = X_test.reshape(10000, 28, 28, 1)

    X_train = X_train.astype('float32')         # change integers to 32-bit floating point numbers
    X_test = X_test.astype('float32')

    X_train /= 255                              # normalize each value for each pixel for the entire vector for each input
    X_test /= 255

    print("Training matrix shape", X_train.shape)
    print("Testing matrix shape", X_test.shape)

    nb_classes = 10 # number of unique digits

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # The Sequential model is a linear stack of layers and is very common.

    model = Sequential()
    # Convolution Layer 1
    model.add(Conv2D(32, (3, 3), input_shape=(28,28,1))) # 32 different 3x3 kernels -- so 32 feature maps
    model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
    convLayer01 = Activation('relu')                     # activation
    model.add(convLayer01)

    # Convolution Layer 2
    model.add(Conv2D(32, (3, 3)))                        # 32 different 3x3 kernels -- so 32 feature maps
    model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
    model.add(Activation('relu'))                        # activation
    convLayer02 = MaxPooling2D(pool_size=(2,2))          # Pool the max values over a 2x2 kernel
    model.add(convLayer02)

    # Convolution Layer 3
    model.add(Conv2D(64,(3, 3)))                         # 64 different 3x3 kernels -- so 64 feature maps
    model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
    convLayer03 = Activation('relu')                     # activation
    model.add(convLayer03)

    # Convolution Layer 4
    model.add(Conv2D(64, (3, 3)))                        # 64 different 3x3 kernels -- so 64 feature maps
    model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
    model.add(Activation('relu'))                        # activation
    convLayer04 = MaxPooling2D(pool_size=(2,2))          # Pool the max values over a 2x2 kernel
    model.add(convLayer04)
    model.add(Flatten())                                 # Flatten final 4x4x64 output matrix into a 1024-length vector

    # Fully Connected Layer 5
    model.add(Dense(512))                                # 512 FCN nodes
    model.add(BatchNormalization())                      # normalization
    model.add(Activation('relu'))                        # activation

    # Fully Connected Layer 6                       
    model.add(Dropout(0.2))                              # 20% dropout of randomly selected nodes
    model.add(Dense(10))                                 # final 10 FCN nodes
    model.add(Activation('softmax'))                     # softmax activation
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # data augmentation prevents overfitting by slightly changing the data randomly
    # Keras has a great built-in feature to do automatic augmentation
    gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)

    test_gen = ImageDataGenerator()
    # We can then feed our augmented data in batches
    # Besides loss function considerations as before, this method actually results in significant memory savings
    # because we are actually LOADING the data into the network in batches before processing each batch

    # Before the data was all loaded into memory, but then processed in batches.

    train_generator = gen.flow(X_train, Y_train, batch_size=128)
    test_generator = test_gen.flow(X_test, Y_test, batch_size=128)

    # We can now train our model which is fed data by our batch loader
    # Steps per epoch should always be total size of the set divided by the batch size

    # SIGNIFICANT MEMORY SAVINGS (important for larger, deeper networks)

    model.fit_generator(train_generator, steps_per_epoch=60000//128, epochs=5, verbose=1, 
                        validation_data=test_generator, validation_steps=10000//128)

    score = model.evaluate(X_test, Y_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


    # The predict_classes function outputs the highest probability class
    # according to the trained classifier for each input example.
    predicted_classes = model.predict_classes(X_test)

    # Check which items we got right / wrong
    correct_indices = np.nonzero(predicted_classes == y_test)[0]

    incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

    plt.figure()
    for i, correct in enumerate(correct_indices[:9]):
        plt.subplot(3,3,i+1)
        plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
        
    plt.tight_layout()
        
    plt.figure()
    for i, incorrect in enumerate(incorrect_indices[:9]):
        plt.subplot(3,3,i+1)
        plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
        
    plt.tight_layout()

