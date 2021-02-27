
# first neural network with keras make predictions
from numpy import loadtxt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
import datetime

NAME = "hole-present-"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir='logs\\fit\\'+NAME, histogram_freq=1)

# load the dataset
dataset = loadtxt('../data/holes6.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:6]
y = dataset[:,7]

train_dataset = tf.data.Dataset.from_tensor_slices((X, y))
# Shuffle and slice the dataset.
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# define the keras model
model = Sequential()
model.add(Dense(20, input_dim=6, activation='relu'))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))
# compile the keras model
print("Compiling model...")
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
"Done."

# fit the keras model on the dataset
print("Fitting model...")
history = model.fit(X, y, epochs=20000, batch_size=20, verbose=0, callbacks=[tensorboard_callback])
"Done."


# make class predictions with the model
print("Making predictions...")
predictions = model.predict(X)
