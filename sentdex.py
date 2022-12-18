import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# print(tf.__version__)

minst=tf.keras.datasets.mnist #28x28 image of hand written digits 0-9
# load data from minst and split into train and test data
(x_train,y_train),(x_test,y_test)=minst.load_data()

# normalize the number
x_train=keras.utils.normalize(x_train,axis=1)
x_test=keras.utils.normalize(x_test,axis=1)

print("start model")
# created model squ

# The first line creates a Sequential model This is the simplest kind of Keras model
model=keras.models.Sequential()
# we build the first layer and add it to the model. It is a Flatten layer whose role is simply to convert each input image into a 1D array
model.add(keras.layers.Flatten())
#hidden layer one and 128 nodes or neuron and activate functions relu
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
#hidden layer two /// /////////////////
model.add(keras.layers.Dense(128,activation=tf.nn.relu))
# Finally, we add a Dense output layer with 10 neurons (one per class), using the softmax activation function (because the classes are exclusive).
model.add(keras.layers.Dense(10,activation=tf.nn.softmax))

# compile:  method to specify the loss function and the optimizer to use
# sgd - adam
model.compile(
               optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"]
              )
history=model.fit(x_train,y_train,epochs=3)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

# # evaluate model
val_loss,val_acc=model.evaluate(x_test,y_test)

print(f"loss is {val_loss} accuracy {val_acc}\n","*"*40)

print("end model")




# plt.imshow(x_train[0], cmap=plt.cm.binary)
# print(x_train[0])

# plt.show()


# #save moedel

# model.save('epic_num_reader_model')
# new_model=tf.keras.models.load_model('epic_num_reader_model')
# #predicts
# predictions=new_model.predict([x_test])

# print(predictions)

# print(np.argmax(predictions[0]))

# plt.imshow(x_test[0])
# print(x_train[0])

# plt.show()

