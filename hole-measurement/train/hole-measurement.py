
# first neural network with keras make predictions
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
import datetime
from sklearn.model_selection import train_test_split

layer1_sizes = [16,]
layer2_sizes = [16,32]
layer3_sizes = [8,16]
layer4_sizes = [8]
test_portion = 0.1

# load the dataset
dataset = np.loadtxt('hole-measurement/data/holes6.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:6]
y = dataset[:,8] / 6.0
print("X shape")
print(X.shape)
XTraining, XValidation, YTraining, YValidation = train_test_split(X,y,stratify=y,test_size=0.1) # before model building
print("X Training shape")
print(XTraining.shape)
print("X Validation shape")
print(XValidation.shape)


#print(y)
#ds_rows = X.shape[0]
#print("Number of rows: {}".format(ds_rows))
#train_rows = round(X.shape[0] * (1-test_portion))
#print("Number of rows in training set: {}".format(train_rows))
#all_ds = tf.data.Dataset.from_tensor_slices((X, y))
#all_ds = all_ds.shuffle(buffer_size=1024)
#train_ds = all_ds.take(train_rows)
#val_ds = all_ds.skip(train_rows)


for layer1_size in layer1_sizes:
    for layer2_size in layer2_sizes:
        for layer3_size in layer3_sizes:
            for layer4_size in layer4_sizes:
                NAME = "hole-size-not-proj-all-relu-normalized-{}-{}-{}-{}-".format(layer1_size, layer2_size,layer3_size, layer4_size) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                tensorboard_callback = TensorBoard(log_dir='logs\\fit\\'+NAME, histogram_freq=1)
                print("Iteration: " +NAME)

                # define the keras model
                model = Sequential()
                model.add(Dense(layer1_size, input_dim=6, activation='relu'))
                model.add(Dense(layer2_size, activation='relu'))
                model.add(Dense(layer3_size, activation='relu'))
                model.add(Dense(layer4_size, activation='relu'))
                model.add(Dense(1, activation='relu'))
                # compile the keras model
                print("Compiling model...")
                model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
                print("Done.")

                # fit the keras model on the dataset
                print("Fitting model...")
                history = model.fit(XTraining, YTraining, validation_data=(XValidation, YValidation), epochs=1000, verbose=0, callbacks=[tensorboard_callback])
                print("Done.")


                # make class predictions with the model
                #print("Making predictions...")
                #predictions = model.predict(X)
                #print("Done.")
