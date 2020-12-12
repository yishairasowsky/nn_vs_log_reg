import sys
from simple_dnn import *

dm = DataManager()
dm.get_data()

print(dm.X.shape)
print(dm.Y.shape)

mm = ModelManager(dm.X_train, dm.Y_train)
mm.train()


db = DecisionBoundary(dm.X_train,dm.Y_train,dm.X_test,dm.Y_test,mm.model)
db.plot()


nn = NeuralNetwork()
a,b,c = nn.layer_sizes(dm.X,dm.Y)
print(a,b,c)

parameters = nn.get_initial_parameters(a,b,c)
A2, cache = nn.forward_propagation(dm.X,parameters)
cost = nn.get_cost(A2,dm.Y,parameters)
print(cost)

grads = nn.backward_propagation(parameters,cache,dm.X,dm.Y)

parameters = nn.update_parameters(parameters, grads, learning_rate=1.2)

nn.nn_model(dm.X, dm.Y, n_h=b, num_iterations=100)

# print(parameters)


