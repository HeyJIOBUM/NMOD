from NMOD_lab3.layer import Layer
from fully_connected_nn import FullyConnectedNN
from keras.api.layers import Dense
from keras.api.models import Sequential
from keras.api.activations import relu

nn = FullyConnectedNN()
nn.add(Layer())

