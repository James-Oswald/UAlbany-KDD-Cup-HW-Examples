#James Oswald 5/23/21
#See part3WithComments.py for a real in depth walkthrough 

#see part3WithComments.py
import tensorflow as tf
from tensorflow import keras                        
from tensorflow.keras.datasets import mnist   
from tensorflow.keras.layers import Dense, Input

#Data loading and processing
#see part3WithComments.py
(trainingImages, trainingLabels), (testingImages, testingLabels) = mnist.load_data()
categoricalTrainingLabels = keras.utils.to_categorical(trainingLabels, 10)
categoricalTestingLabels = keras.utils.to_categorical(testingLabels, 10)
flatTrainingImages = tf.reshape(trainingImages, shape=[-1, 784])
flatTestingImages = tf.reshape(testingImages, shape=[-1, 784])

#A custom training loop can greatly be simplified using tensorflows dataset class
#and its built in iterators and batching. This will also greatly simplify our 
#  We turn our processed data into a tensorflow
#dataset using tf.data.Dataset.from_tensor_slices() see:
#https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices
#We then can use the dataset's .batch() method to automatically batch for us
#and do random shuffeling behind the scenes when we iterate later in the loop
trainingDataset = tf.data.Dataset.from_tensor_slices((flatTrainingImages, categoricalTrainingLabels))
trainingDataset = trainingDataset.batch(32)

#Model Creation
#see part3WithComments.py
model = keras.Sequential([
    Input(784), 
    Dense(30, "sigmoid"), 
    Dense(10, "sigmoid")
])

#we're only comiling so we can call model.evaluate later with our testing data
#we could also do a custom testing loop as well and never compile
model.compile(metrics=["accuracy"])

#model training with a custom training loop
#see https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
#We no longer are compiling the model, we need to set the loss function and optimizer ourselves
optimizer = keras.optimizers.SGD()  #Stochastic Gradient Descent
loss = keras.losses.MSE             #Mean Squared Error

#The main training loop
#this is heavily influced by https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
#The core idea of epochs is that each epoch we go through every single training example
#in the training dataset, we specify 10 epochs.
for epoch in range(10): 
    print("\nStart of epoch " + str(epoch))

    #In each epoch we go through all batches in trainingDataset
    #we update the weights after each batch using auto grad.
    for trainingImageBatch, trainingLabelBatch in trainingDataset:

        #The core idea of automatic differentiation is that we
        #"watch" the forward pass results with respect to the weights,
        #the effects are recorded on the tape whose results are used by
        #the optimizer to change the weights
        with tf.GradientTape() as tape:
            #forward pass a whole batch through the model
            #the shape of trainingImageBatch is 32 by 784, the output is
            #of shape 32 by 10, the 10 final outputs of the network for each sample
            lastLayerOutputBatch = model(trainingImageBatch, training=True)
            #we compute the 32 MSE losses for each item in the batch vs
            #its expected output, trainingLabelBatch of shape 32 by 10.
            #batchLosses is of shape 32 by 1, as each item in the batch only has 1 loss 
            batchLosses = loss(lastLayerOutputBatch, trainingLabelBatch)

        #lmao backprop in one line?
        #retrive the gradients on the loss with respect to the models trainable params.
        grads = tape.gradient(batchLosses, model.trainable_weights)

        #update the weights and biases in accordance with SGD using our gradients
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

#model evaluation on unseen data
#see part3WithComments.py
score = model.evaluate(flatTestingImages, categoricalTestingLabels)
print("Testing Accuracy: %" + str(100*score[1]))

#This takes way longer to run because more code is in python rather then C,
#we expect a testing accuracy around 87%.