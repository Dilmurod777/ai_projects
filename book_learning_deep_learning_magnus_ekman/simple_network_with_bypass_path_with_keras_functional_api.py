from keras.layers import Input, Dense, Concatenate
from keras.models import Model

# declare inputs
inputs = Input(shape=(10,))
bypass_inputs = Input(shape=(5,))

# declare layers
layer1 = Dense(64, activation='relu')
concat_layer = Concatenate()
layer2 = Dense(64, activation='relu')

# connect inputs and layers
layer1_outputs = layer1(inputs)
layer2_inputs = concat_layer([layer1_outputs, bypass_inputs])
layer2_outputs = layer2(layer2_inputs)

# create model
model = Model(inputs=[inputs, bypass_inputs],
							outputs=layer2_outputs)
model.summary()
