# Testing a simple neuronal network usiong tensorflow and keras
# We will try to deduce the algorithm to convert C degrees into F degrees
# Fahrenheit = Celcius * 1.8 + 32
# David Rodriguez

# import modules
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# learning inputs
celsius = np.array([-40,10,0,8,15,22,38], dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100], dtype=float)

# we create a layer (output neurons, input neurons) (1 Layer)
layer = tf.keras.layers.Dense(units=1, input_shape=[1])

'''
# we create a layer (output neurons, input neurons) (3 Layers)
layer1 = tf.keras.layers.Dense(units=3, input_shape=[1])
layer2 = tf.keras.layers.Dense(units=3)
exit_layer = tf.keras.layers.Dense(units=1)
'''

# we define a sequential model
model = tf.keras.Sequential(layer)

# we define a compiler
model.compile(

    #We selected Adam as optimizer (learning ratio)
    optimizer = tf.keras.optimizers.Adam(0.1),
    # we define mse as loss function
    loss = 'mean_squared_error'
) 


print("Ininitializing learning...")

#we use the fit function to initialize (inputs, outputs, iterations)
record = model.fit(celsius,fahrenheit, epochs=1000, verbose=False)

print("Model trained!")

# We display the results
# Hint: Use a jupyter notebook to display the graphs
plt.xlabel("# Epoch")
plt.ylabel("Loss magnitude")
plt.plot(record.history["loss"])


# Unit tests
input_value = float(100)
print("-----Tests-----\n")

#res = (input_value*weight)+bias
res = model.predict([input_value])
print(f"Expected: {input_value*1.8+32} = {input_value} * 1.8 + 32")
print(f"Result: {str(res[0][0])} = {input_value} * {layer.get_weights()[0][0][0]} + {layer.get_weights()[1][0]}\n")
print(f'''Internal model values: 
Weight:{str(layer.get_weights()[0][0][0])}
Bias: {str(layer.get_weights()[1][0])}''')


