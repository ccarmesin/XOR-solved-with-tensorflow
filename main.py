import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

# Define training data
xs = tf.constant([[0, 0],[0, 1],[1, 0],[1, 1]], tf.float32)
ys = tf.constant([[0], [1], [1], [0]], tf.float32)

# Hyperparameters
model = Sequential()
epochs = 10
batchSize = 4
stepsPerEpoch = 10

loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(.3)

# Model architecture
inputNodes = 2
hiddenNodes = 4
outputNodes = 1

def modelBuild():

    # Define the keras model
    model.add(Dense(hiddenNodes, activation='tanh', input_dim=inputNodes))
    model.add(Dense(outputNodes, activation='tanh'))

    # Compile the keras model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

def train(xs, ys):

    # Fit the keras model on the dataset
    model.fit(xs, ys, epochs=epochs, batch_size=batchSize, steps_per_epoch=stepsPerEpoch, shuffle=True)

def evaluate(xs, ys):

    # Evaluate the just trained model
    lossValue, accuracy = model.evaluate(xs, ys, steps=4)
    print('Accuracy: %.2f' % (accuracy * 100), '\nLoss: ', lossValue)

if __name__ == "__main__":

    modelBuild()
    train(xs, ys)
    evaluate(xs, ys)
