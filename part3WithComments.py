#James Oswald 5/23/21
#I hope you like reading lmao.

#TensorFlow is the core library it conatins low level tools 
#and classes for building powerful machine learning models
import tensorflow as tf

#Keras is a high level library packaged with TensorFlow that 
#gives us access to high level model building tools and sample datasets
from tensorflow import keras                        #For General Keras utils we'll use
from tensorflow.keras.datasets import mnist         #Our sample dataset
from tensorflow.keras.layers import Dense, Input  #Layer classes for building our network

#Data from mnist.load_data() is provided in the shape of a tuple of tuples containing:
#   ((trainImages, trainLabels), (testImages, testLabels))
# we unpack this over multiple lines:
trainingData, testingData = mnist.load_data()
trainingImages, trainingLabels = trainingData
testingImages, testingLabels = testingData

#labels come in the form of the literal number they coresond to as an image:
#   for example: if trainingImages[0] is an image of a 5, trainingLabels[0] == 5 
#keep in mind we need to represent the label as the desired output of the network:
#   for example: 5 would need to be represented as [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
#Keras has a built in util function that lets us transform our labels into this form.
#see https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
categoricalTrainingLabels = keras.utils.to_categorical(trainingLabels, 10)
categoricalTestingLabels = keras.utils.to_categorical(testingLabels, 10)

#our training images are 28 by 28 pixel grayscale (one float per pixel) images,
#but we want to feed them to our network as one long 784 long array. To do this
#We need to "Flatten" Them, normally we do this with a flattening layer in the model
#but here I have decided to pre-flatten the image data with tf.reshape()

#our input data has the shape of a big loaf of bread (numImages, 28, 28)
#numImages is the number of bread slices (28 by 28 pixel images) we have
#tf.reshape takes a tensor (3d array) and reshapes it into a new shape
#that we specify. I want the new shape to be (numImages, 784) IE,
#turn each of our 28 by 28 slices into one long 784 by 1 array 
#To acchive this I pass in [-1, 784] as the new shape, -1 specifies
#that we would like to copy the length of the original dimention here (numImages) 
#thus these equivilent if we know numImages off the top of our head:
#   tf.reshape(x, shape=[-1, 784]) == tf.reshape(x, shape=[numImages, 784])
#For more see: https://www.tensorflow.org/api_docs/python/tf/reshape

flatTrainingImages = tf.reshape(trainingImages, shape=[-1, 784])
flatTestingImages = tf.reshape(testingImages, shape=[-1, 784])

#next we build our model, Keras is here for that too. Keras provides a high level
#model building API that lets us easily construct models just like we've been learning
#about. keras.Sequential is a class used to contain layers. It represents an array
#of sequential layers, hence the name.
#

#The Input layer is the neurons we set to value of the initial data being fed in,
#remember that input neurons are NOT perceptrons, rather we literally set their 
#output values to the data. We pass in the shape (number of nodes in the input layer)
#https://www.tensorflow.org/api_docs/python/tf/keras/Input
# ^ Just pretend this is a layer, don't look too deep into it, I never have and I'm doing ok.

#The Dense layers are the standard sigmoid perceptron layers we have been looking at.
#we can pass in the size of the layer as the first parameter, and also what activation
#function we want used, we select sigmoid to make them sigmoid neurons.
#https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
model = keras.Sequential([
        Input(shape=784),                   #remember the input layer is NOT a perceptron layer
        Dense(30, activation="sigmoid"),    #Dense layers are our nice perceptron layers
        Dense(10, activation="sigmoid"),    #output layers are also perceptrons
])

#We "compile" the model to some super fast GPU accelerated graph program by specifying
#our loss function (Mean squared error, remeber 3blue1brown vids!) and using stocastic
#gradient descent (SGD) as our optimizer.
#https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
model.compile(
    loss="MSE",            #Mean squred error loss, see https://youtu.be/IHZwWFHWa-w?t=230
    optimizer="SGD",       #stochastic gradient descent, (aka how we update the weights & biases)
    metrics=["accuracy"]   #we would like to track % of images we correctly classify.
)

#next we "fit" (run) the compiled model. We pass it the properly formatted
#training data and training labels as well as the batch size and number of epochs.
#https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
model.fit(
    x=flatTrainingImages,          #The network inputs: 28 by 28 images flattened to 784 by 1
    y=categoricalTrainingLabels,   #The expected network outputs: catagorial labels represnting numbers
    batch_size=30,                 #The batch size, amount of samples to look at before updating the weights & biases
    epochs=10                      #Number of times to pass through all training data.
)

#finally we evaluate the model on data it has never seen before, this is the true test 
#to make sure it hasn't just memorized the training data (aka overfitting). 
score = model.evaluate(
    x=flatTestingImages,        #see above
    y=categoricalTestingLabels  #see above
)

#if we run this file we should see ~80% training and testing accuracy.

