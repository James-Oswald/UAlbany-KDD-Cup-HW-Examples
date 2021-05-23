#James Oswald 5/23/21
#See part3WithComments.py for a real in depth walkthrough 

import tensorflow as tf
from tensorflow import keras                        
from tensorflow.keras.datasets import mnist   
from tensorflow.keras.layers import Dense, Input

#Data loading and processing
(trainingImages, trainingLabels), (testingImages, testingLabels) = mnist.load_data()
categoricalTrainingLabels = keras.utils.to_categorical(trainingLabels, 10)
categoricalTestingLabels = keras.utils.to_categorical(testingLabels, 10)
flatTrainingImages = tf.reshape(trainingImages, shape=[-1, 784])
flatTestingImages = tf.reshape(testingImages, shape=[-1, 784])

#Model Creation
model = keras.Sequential([
    Input(784), 
    Dense(30, "sigmoid"), 
    Dense(10, "sigmoid")
])
model.compile(loss="MSE", optimizer="SGD", metrics=["accuracy"])

#model training and testing 
model.fit(flatTrainingImages, categoricalTrainingLabels, 30, 10)
score = model.evaluate(flatTestingImages, categoricalTestingLabels)
print("Testing Accuracy: %" + str(100*score[1]))