#James Oswald 5/23/21
#See part3WithComments.py for a real in depth walkthrough 

import tensorflow as tf
from tensorflow import keras                        
from tensorflow.keras.datasets import mnist   
from tensorflow.keras.layers import Dense, Input

#Data loading and processing, we have more to do here 
(trainingImages, trainingLabels), (testingImages, testingLabels) = mnist.load_data()
categoricalTrainingLabels = keras.utils.to_categorical(trainingLabels, 10)
categoricalTestingLabels = keras.utils.to_categorical(testingLabels, 10)
flatTrainingImages = tf.reshape(trainingImages, shape=[-1, 784])
flatTestingImages = tf.reshape(testingImages, shape=[-1, 784])

trainingDataset = tf.data.Dataset.from_tensor_slices((flatTrainingImages, categoricalTrainingLabels))
trainingDataset = trainingDataset.batch(32)

#Model Creation
model = keras.Sequential([
    Input(784), 
    Dense(30, "sigmoid"), 
    Dense(10, "sigmoid")
])
model.compile(metrics=["accuracy"])

#model training with a custom training loop
optimizer = keras.optimizers.SGD()
loss = keras.losses.MSE
for epoch in range(10): 
    print("\nStart of epoch " + str(epoch))
    for trainingImageBatch, trainingLabelBatch in trainingDataset:
        with tf.GradientTape() as tape:
            lastLayerOutputBatch = model(trainingImageBatch, training=True)
            batchLosses = loss(lastLayerOutputBatch, trainingLabelBatch)
        grads = tape.gradient(batchLosses, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

score = model.evaluate(flatTestingImages, categoricalTestingLabels)
print("Testing Accuracy: %" + str(100*score[1]))