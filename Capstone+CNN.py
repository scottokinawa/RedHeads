
# coding: utf-8

# I am going to write down my code for my image classifier using Keras. I think AWS already does a facial recognizer for you, but I wanted to go ahead and write my own code to identify myself from others, hopefully. I want to see if I can get a regular CNN to work closely to that of a facial recognizer. Lets begin an find out!

# In[10]:

#imports
from keras.models import Sequential 
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


# In[2]:

classifier = Sequential()


# Step 1 - Convolution 
# 3 by 3 dimentions. 128 feauture maps will be 
# made. Input_shape() is important because we need to reformat everythig to have
# shapes. We are working with colored data so we are going to use 3d array.
# Which is the last number in input_shape() you will see 64, 64 first because
# this is tensor flow, we normally put something higher like 128 or even 256 
# because that will leave better results, but becuase we are working with 
# a cpu again we are going to use a smaller amount. We will use relu so we 
# don't have linearity and don't have any negative numbers.

# In[3]:

classifier.add(Convolution2D(128,(3,3),input_shape=(128,128,3), activation = 'relu'))


# Step 2 - Pooling
# We slide the 2x2 sub table and slide over the feature map and take the max
# of the four cells. The pooling layer contains all these new maps. We want to 
# reduce our feature maps even more during the pooling layer to decrease our 
# nodes  which decreases the fully connected layer which helps make everything 
# run less computationally.

# In[4]:

classifier.add(MaxPooling2D(pool_size = (2,2)))


# In[5]:

# Adding a second convolutional layer
classifier.add(Convolution2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[6]:

# Adding a third convolutional layer 
classifier.add(Convolution2D(256, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# #Step 3

# In[7]:

classifier.add(Flatten())


# Step 4 - Full Conection 
# 
# Making a classic ANN of layers. Use this input layer made from flattening
# and then make a hidden layer and an binary output layer because cat or dog. 
# Dense(ouput_dim = amoutn of nodes. We don't know how many to add so lets make
# a good guess on what to put. 128, is from experimenation. Not too small to 
# decrease accuracy but not too high bec of computation costs. Last parameter
# lets add the activation function relu.

# In[8]:

classifier.add(Dense(output_dim = 200, activation = 'relu'))


# In[11]:

classifier.add(Dropout(.5))


# Step 5 - output later 
# 
# Just copy the layer above but switch to the sigmoid bec binary. Also switch
# the output_dim to 1 because one thing we want to check, Cats or Dogs.

# In[12]:

classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) 


# Step 6 - Compiling the CNN
# 
# Here we chose adam as our optimizer because it is great stochastic optimizer.
# We chose binary_crossentropy because it is a sigmoid function. However, if it 
# were more than one output, we would probably choose categorical_crossentropy.

# In[13]:

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics =['accuracy'])


# # Part 2 

# In[15]:

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                '/Users/Scott/Documents/DSI6/FaceRecognition/pictures/train',
                                                target_size=(128, 128),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                            '/Users/Scott/Documents/DSI6/FaceRecognition/pictures/test',
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                    samples_per_epoch=8000,
                    nb_epoch=25,
                    validation_data=test_set,
                    validation_steps=2000)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



