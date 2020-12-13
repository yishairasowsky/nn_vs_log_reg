from simple_dnn import DataManager, ModelManager, DecisionBoundary, NeuralNetwork

dm = DataManager()
dm.get_data()

mm = ModelManager(dm.X_train, dm.Y_train)
mm.train()

db = DecisionBoundary(dm.X_train,dm.Y_train,dm.X_test,dm.Y_test,mm.model)
db.plot()

nn = NeuralNetwork(num_iterations=3000,num_hidden_units=5)
nn.evaluate(dm.X_train,dm.Y_train,dm.X_test,dm.Y_test)