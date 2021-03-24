# Lab 3
# Implement in Python + Numpy
# a Neural Network ( sigmoid( W0 * sigmoid( W1 * x ) ) )
# + Gradient Descent.

import pickle
import numpy as np

with open('data.pkl', 'rb') as f:
  data = pickle.load(f)
training_data, test_data = data[0], data[2]

np.random.seed(1000)

training_data = np.asarray(training_data)
test_data = np.asarray(test_data)

n_input, n_hidden, n_output = 784, 100, 10
biases = [ np.random.randn(n_hidden, 1), np.random.randn(n_output, 1) ]
weights = [ np.random.randn(n_hidden, n_input), np.random.randn(n_output, n_hidden) ]

n_epochs, lr = 80, 12

# TODO: implement a function which calculates the sigmoid / derivative of the sigmoid function
def sigmoid(z, deriv = False):
  if deriv:
    return (1/(1+ np.exp(-z)))*(1-(1/(1+ np.exp(-z))))
  else:
    return 1/(1+ np.exp(-z))

# TODO: implement forward pass
def forward(x):
  wxb0 = weights[0].dot(x) + biases[0] #wejsciowy * macierz wag +bias
  hidden = sigmoid(wxb0) #sigmoid od ^
  wxb1 = weights[1].dot(hidden) + biases[1] #hidden *macierz wag + bias
  output = sigmoid(wxb1) #sigmoid wxb1
  return wxb0, hidden, wxb1, output


# TODO: implement backprop
def backprop(x, y):
  nabla_b = [np.zeros(biases[0].shape), np.zeros(biases[1].shape)]
  nabla_w = [np.zeros(weights[0].shape), np.zeros(weights[1].shape)]

  # forward pass
  wxb0, hidden, wxb1, output = forward(x)

  # array([[1,2,3],]*3).transpose()

  # backward pas
  der_j_wrt_o = output - y
  der_o_wrt_wxb1 = sigmoid(wxb1, deriv=True)
  der_wxb1_wrt_h = weights[1]
  der_h_wrt_wxb0 = sigmoid(wxb0, deriv=True)

  nabla_b[1] = np.multiply(der_j_wrt_o, der_o_wrt_wxb1)
  nabla_w[1] = (np.multiply(der_j_wrt_o, der_o_wrt_wxb1)).dot(hidden.T)
  nabla_b[0] = np.multiply((np.multiply(der_j_wrt_o, der_o_wrt_wxb1)).T.dot(weights[1]).T, (der_h_wrt_wxb0))
  nabla_w[0] = np.multiply((np.multiply(der_j_wrt_o, der_o_wrt_wxb1)).T.dot(weights[1]).T, (der_h_wrt_wxb0)).dot(x.T)
  return nabla_w, nabla_b


# TODO: train + evaluate
for ep in range(n_epochs):
  # train
  nabla_w = [np.zeros(weights[0].shape), np.zeros(weights[1].shape)]
  nabla_b = [np.zeros(biases[0].shape), np.zeros(biases[1].shape)]
  for x, y in training_data:
    nabla_wi, nabla_bi = backprop(x, y)
    nabla_w = [nw + nwi for nw, nwi in zip(nabla_w, nabla_wi)]
    nabla_b = [nb + nbi for nb, nbi in zip(nabla_b, nabla_bi)]
  weights = [weights[0] - lr * (nabla_w[0] / len(training_data)), weights[1] - lr * (nabla_w[1] / len(training_data))]
  biases = [biases[0] - lr * (nabla_b[0] / len(training_data)), biases[1] - lr * (nabla_b[1] / len(training_data))]

  # evaluate
  s = 0
  for x, y in test_data:
    _, _, _, output = forward(x)
    s += int(np.argmax(output) == y)
  print("Epoch {} : {} / {}".format(ep, s, len(test_data)))

