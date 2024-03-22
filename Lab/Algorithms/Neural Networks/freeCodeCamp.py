from __future__ import absolute_import, division, print_function, unicode_literals 
"""
Created on Mon Oct 11 13:28:04 2021

@author: juanbeta
"""

'''
https://www.youtube.com/watch?v=tPYj3fFJGjk
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf


#%%#############     Introduction    ################# 
# Rank 0 Tensors, data types
string = tf.Variable('This is a string', tf.string)
number = tf.Variable(323, tf.int16)
floating = tf.Variable(3.567, tf.float64)

# Rank 1 and 2 Tensors
rank1_tensor = tf.Variable(['Test', 'Ok', 'Tim'], tf.string)
rank2_tensor = tf.Variable([['Test', 'Ok'], ['test', 'yes']], tf.string)

# Rank and shape
tf.rank(rank2_tensor)
rank2_tensor.shape

# Changing shape
tensor1 = tf.ones([1,2,3])
tensor2 = tf.reshape(tensor1, [2,3,1])
tensor3 = tf.reshape(tensor2, [3,-1])       # -1 indicates to TF to adjust that dimention to fit the data


# Types of Tensors
'''
Variable 
Constant 
'''

#%%#############  Linear Regression  #################


'''
https://colab.research.google.com/drive/15Cyy2H7nT40sGR7TBN5wBvgTd57mVKay#forceEdit=true&sandboxMode=true&scrollTo=d0dfaT4esRh3
'''

# Load dataset
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
dftrain.head()
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

dftrain.describe()
dftrain.shape

# Cualitative analysis
# dftrain.age.hist(bins=20)
# dftrain.sex.value_counts().plot(kind='barh')
# dftrain['class'].value_counts().plot(kind='barh')
# pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')

# Categorical data
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

# Input function
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# Create model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# We create a linear estimtor by passing the feature columns we created earlier

# Train the model
linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

clear_output()  # clears consoke output
print(f'Accuracy: {result["accuracy"]}')  # the result variable is simply a dict of stats about our model

# Test the model 
result = list(linear_est.predict(eval_input_fn))
print(result[0]['probabilities'])


#%%#############   Classification    #################


'''
https://colab.research.google.com/drive/15Cyy2H7nT40sGR7TBN5wBvgTd57mVKay#forceEdit=true&sandboxMode=true&scrollTo=d0dfaT4esRh3
'''

#Load data
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# Lets define some constants to help us later on

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
# Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe

# Adequation of data
train_y = train.pop('Species')
test_y = test.pop('Species')
train.head() # the species column is now gone

# Input function
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

# Feature columns describe how to use the input.
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3)

# Train the model
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)
# We include a lambda to avoid creating an inner function previously

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# Predict
def input_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
    valid = True
    while valid: 
        val = input(feature + ": ")
        if not val.isdigit(): valid = False
    predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))

'''
serio = input('Socio cómo se llama gonorrea?')
print(serio + ' Como es perro hijueputa')
'''


#%%#############        MDP's        #################


import tensorflow_probability as tfp  # We are using a different module from tensorflow this time

tfd = tfp.distributions  # making a shortcut for later on
initial_distribution = tfd.Categorical(probs = [0.2, 0.8])  # Refer to point 2 above
transition_distribution = tfd.Categorical(probs = [[0.5, 0.5],
                                                  [0.2, 0.8]])  # refer to points 3 and 4 above
observation_distribution = tfd.Normal(loc = [0., 15.], scale = [5., 10.])  # refer to point 5 above
# the loc argument represents the mean and the scale is the standard devitation

model = tfd.HiddenMarkovModel(
    initial_distribution = initial_distribution,
    transition_distribution = transition_distribution,
    observation_distribution = observation_distribution,
    num_steps = 7) 

mean = model.mean()

# due to the way TensorFlow works on a lower level we need to evaluate part of the graph
# from within a session to see the value of this tensor

# in the new version of tensorflow we need to use tf.compat.v1.Session() rather than just tf.Session()
with tf.compat.v1.Session() as sess:  
  print(mean.numpy())


#%%#############    Neural Netwoks   #################


'''
https://colab.research.google.com/drive/1m2cg3D1x3j5vrFc-Cu0gMvc48gWyCOuG#forceEdit=true&sandboxMode=true&scrollTo=-HJV4JF789aC
'''

from tensorflow import keras

# Data set
fashion_mnist = keras.datasets.fashion_mnist  # load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into tetsing and training

train_images.shape 
train_images[0,23,23]

train_labels[:10]               # Train labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure()                        # Images to clasify
plt.imshow(train_images[2])
plt.colorbar()
plt.grid(False)
plt.show()

# !!! Data Preprocessing
train_images = train_images / 255.0             # Normalizing the input
test_images = test_images / 255.0

# Building the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),     # input layer (1)
    keras.layers.Dense(128, activation = 'relu'),   # hidden layer (2)
    keras.layers.Dense(10, activation = 'softmax')  # output layer (3)
])

# Compile the model
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs = 8)  # we pass the data, labels and epochs and watch the magic!

# Testing the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose = 1) 

# print('Test accuracy:', test_acc)

# Making predictions
predictions = model.predict(test_images)
# print(class_names[np.argmax(predictions[1000])])
# plt.figure()                        # Images to clasify
# plt.imshow(test_images[1000])
# plt.colorbar()
# plt.grid(False)
# plt.show()

COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)


#%%#######  Convolutional Neural Netwoks  ############


'''
https://colab.research.google.com/drive/1ZZXnCjFEOkp_KdNcNabd14yok0BAIuwS#forceEdit=true&sandboxMode=true
'''




