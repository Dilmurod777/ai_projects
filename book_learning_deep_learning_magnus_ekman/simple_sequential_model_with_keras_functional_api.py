from keras.layers import Input, Dense
from keras.models import Model

# declare inputs
inputs = Input(shape=(10,))

# declare layers
layer1 = Dense(64, activation='relu')
layer2 = Dense(64, activation='relu')

# connect inputs and layers
layer1_outputs = layer1(inputs)
layer2_outputs = layer2(layer1_outputs)

# create model
model = Model(inputs=inputs, outputs=layer2_outputs)
model.summary()
