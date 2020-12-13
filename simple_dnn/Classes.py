import numpy as np
import matplotlib.pyplot as plt

from numpy import arange, meshgrid, hstack
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

np.random.seed(1) # set a seed so that the results are consistent

noise=0.15
factor=0.5
n_samples = 1000


class DataManager:

    def __init__(self):
        pass


    def get_data(self):

        data = datasets.make_moons(n_samples=n_samples, noise=noise)

        X, Y = data
        
        self.X = X.T
        self.Y = Y.reshape(1,Y.shape[0])

        test_size=0.33


        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X.T, 
        
        self.Y.T, test_size=test_size, random_state=42)
        
        self.X_train=self.X_train.T
        self.X_test=self.X_test.T
        self.Y_train=self.Y_train.T
        self.Y_test=self.Y_test.T

        plt.scatter(self.X_train[0, :], self.X_train[1, :], c=self.Y_train, s=5, cmap=plt.cm.spring)

        plt.scatter(self.X_test[0, :], self.X_test[1, :], c=self.Y_test, s=20, cmap=plt.cm.spring, marker='*')

        plt.savefig('data')


class ModelManager:

  def __init__(self, X_train, Y_train):
    self.X_train = X_train
    self.Y_train = Y_train


  def train(self):
    self.model = LogisticRegression()
    self.model.fit(self.X_train.T, self.Y_train.T.ravel())
    print('done training!')


class DecisionBoundary:

  def __init__(self, X_train, Y_train, X_test, Y_test, model):
    self.X_train = X_train 
    self.Y_train = Y_train
    self.X_test = X_test 
    self.Y_test = Y_test
    self.model = model


  def plot(self):

    min1, max1 = self.X_train[0,:].min()-1, self.X_train[0,:].max()+1
    min2, max2 = self.X_train[1,:].min()-1, self.X_train[1,:].max()+1

    x1grid = arange(min1, max1, 0.1)
    x2grid = arange(min2, max2, 0.1)

    xx, yy = meshgrid(x1grid, x2grid)
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

    grid = hstack((r1,r2))
    yhat = self.model.predict(grid)
    zz = yhat.reshape(xx.shape)

    plt.contourf(xx, yy, zz, cmap='Paired');
    plt.scatter(self.X_train[0, :], self.X_train[1, :], c=self.Y_train, s=10, cmap=plt.cm.Spectral)
    plt.savefig('dec_bdy')
    plt.clf()

    yhat = self.model.predict(self.X_test.T)
    acc = accuracy_score(self.Y_test[0], yhat)
    print(f'Logistic Regression Accuracy: {acc}')


class NeuralNetwork:
  
  def __init__(self,num_iterations,num_hidden_units):

    self.num_iterations = num_iterations
    self.num_hidden_units = num_hidden_units

 
  def layer_sizes(self, X, Y):
 
    n_x = X.shape[0] # input 2 features/sample
 
    n_h = self.num_hidden_units # nodes in hidden layer
 
    n_y = Y.shape[0] # 1 output
 
    return (n_x, n_h, n_y)

 
  def get_initial_parameters(self,n_x, n_h, n_y):
 
      # must be non-zero; o/w the NN won't learn be influenced by the inputs
      # must be different; o/w gradient for all neurons will be same 
      W1 = np.random.randn(n_h, n_x) * 0.01
      W2 = np.random.randn(n_y, n_h) * 0.01
 
      # OK to be all zero
      b1 = np.zeros(shape=(n_h, 1))
      b2 = np.zeros(shape=(n_y, 1))
 
      return W1,b1,W2,b2


  def sigmoid(self,Z):

    return 1/(1 + np.exp(-Z))


  def forward_propagation(self,X, parameters):
    
    W1,b1,W2,b2 = parameters

    Z1 = np.dot(W1, X) + b1 # get linear comb
    A1 = np.tanh(Z1) # get activation value
    
    Z2 = np.dot(W2, A1) + b2
    A2 = self.sigmoid(Z2)
    
    cache = Z1,A1,Z2,A2

    return A2, cache


  def get_cost(self, A2, Y, parameters):

    m = Y.shape[1] # number of examples

    W1,_,W2,_ = parameters

    # prob high when A2 (our prediction) is close to to Y (the true value)
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))

    # negate b/c want low loss for high prob
    cross_entropy = - np.sum(logprobs) / m
    
    # convert matrix to real number
    cost = np.squeeze(cross_entropy) 
    
    return cost


  def backward_propagation(self,parameters, cache, X, Y):
    
    m = X.shape[1]
    W1,_,W2,_ = parameters
    _,A1,_,A2 = cache

    # for chain rule, need partial derivatives of cost
    # e.g. dZ2 is d(cost)/dZ2
    # new param will be old minus rate*deriv_of_old
    dZ2= A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = dW1, db1, dW2, db2
    
    return grads


  def update_parameters(self, parameters, grads, learning_rate=1.2):

    W1,b1,W2,b2 = parameters

    dW1,db1,dW2,db2 = grads

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = W1,b1,W2,b2

    return parameters


  def nn_model(self, X, Y, print_cost=False):

    np.random.seed(3)

    n_x = self.layer_sizes(X, Y)[0]
    n_y = self.layer_sizes(X, Y)[2]
    
    parameters = self.get_initial_parameters(n_x, self.num_hidden_units, n_y)
    W1,b1,W2,b2 = parameters

    for i in range(0, self.num_iterations):

        # calc outputs
        A2, cache = self.forward_propagation(X, parameters)

        # calc error
        cost = self.get_cost(A2, Y, parameters)

        # calc gradient
        grads = self.backward_propagation(parameters, cache, X, Y)

        # update params (weights, biases)
        parameters = self.update_parameters(parameters, grads)

        if print_cost and i % int(self.num_iterations/5) == 0:
            print (f"Cost after iteration {i}: {cost}f")

    return parameters


  def predict(self, parameters, X):

      A2, _ = self.forward_propagation(X, parameters)

      predictions = np.round(A2)

      return predictions


  def evaluate(self, X_train, Y_train, X_test, Y_test):

    # Build a model with a n_h-dimensional hidden layer
    parameters = self.nn_model(X_train, Y_train, print_cost=True)

    yhat = self.predict(parameters, X_test)
    acc = accuracy_score(Y_test.T, yhat.T)
    print("Deep Neural Net Accuracy:",acc)

    plt.scatter(X_test[0, :], X_test[1, :], c=Y_test!=yhat, s=40, cmap=plt.cm.bwr)
    plt.savefig('preds')


def main():
  pass

if __name__ == "__main__":
   main()