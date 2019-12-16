import torch
import numpy as np
from matplotlib import pyplot as plt
from nn_models import *

"""
Hyperparameters
"""
num_iter = 50
learning_rate = 0.0003
batch_size = 32

"""
Read data from the specified training, validation and test data files.
Using the whole image, not creating other features
"""
def read_data(trainFile, valFile, testFile):
    # trian, validation, and test data loader
    #print(use_mlp)
    data_loaders = []

    # read training, test, and validation data
    for file in [trainFile, valFile, testFile]:
        # read data
        data = np.loadtxt(file)

        # digit images
        imgs = torch.tensor(data[:,:-1]).float().cuda()

        # divide each image by its maximum pixel value for numerical stability
        imgs = imgs / torch.max(imgs,dim=1).values[:,None]

        # labels for each image
        labels = torch.tensor(data[:,-1]).long().cuda()

        # if using CNN model, reshape each image:
        # [batch x num_channel x image width x image height]
        if not use_mlp:
            imgs = imgs.view(-1,1,28,28)

        # create dataset and dataloader, a container to efficiently load data in batches
        dataset = utils.TensorDataset(imgs,labels)
        dataloader = utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        data_loaders.append(dataloader)

    return data_loaders[0], data_loaders[1], data_loaders[2]

"""
Train Multilayer Perceptron (MLP)

Initialize MLP model --> define loss function 
                     --> define optimizer
                     --> train model with num_iter epochs 
                     --> pick the best model and return
    - Parameters: train_loader --- the train dataloader
                  val_loader --- the validation dataloader
    - Return:     net --- the best trained MLP network with the lowest validation loss
                  avg_train_loss --- a list of averaged training loss of length num_iter
                  avg_val_loss --- a list of averaged validation loss of length num_iter
"""
def trainMLP(train_loader,val_loader):
    # average training loss, one value per iteration (averaged over all batches in one iteration)
    avg_train_loss = []

    # average validation loss, one value per iteration (averaged over all batches in one iteration)
    avg_val_loss = []

    # record the lowest validation loss, used to determine early stopping (best model)
    best_val_score = float('inf')
    best_iteration = 0
    net = BaselineMLP()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate)

    # define loss function
	# define optimizer
	# for each iteration, iteratively train all batches
	for i in range(0, num_iter):
		print(f"Iteration {i}")

		train_loss = 0
		net.train()
		for point in train_loader:
			data, label = point
			pred_label = net(data)
			loss = loss_func(pred_label, label)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
			train_loss = train_loss / len(train_loader)
		train_loss = train_loss / len(train_loader)
		torch.save(net.state_dict(), f"epoch{i}.pt")
		avg_train_loss.append(train_loss)

		te_loss = 0
		net.eval()
		for point in val_loader:
			data, label = point
			pred_label = net(data)
			loss = loss_func(pred_label, label)
			te_loss += loss.item()
		te_loss = te_loss / len(val_loader)
		avg_val_loss.append(te_loss)

		if te_loss < best_val_score:
			best_val_score = te_loss
			best_iteration = i

	net.load_state_dict(torch.load(f"epoch{best_iteration}.pt"))
	return net, avg_train_loss, avg_val_loss

"""
Train Baseline Convolutional Neural Network (CNN)

Initialize CNN model    --> define loss function 
						--> define optimizer
						--> train your model with num_iter epochs 
						--> pick the best model and return
    - parameters: 	train_loader --- the train dataloader
					val_loader --- the validation dataloader
	- Return:     	net --- the best trained MLP network with the lowest validation loss
                  	avg_train_loss --- a list of averaged training loss of length num_iter
                  	avg_val_loss --- a list of averaged validation loss of length num_iter

"""
def trainCNN(train_loader,val_loader):
    # average training loss, one value per iteration (averaged over all batches in one iteration)
    avg_train_loss = []

    # average validation loss, one value per iteration (averaged over all batches in one iteration)
    avg_val_loss = []

    # record the lowest validation loss, used to determine early stopping (best model)
    best_val_score = float('inf')
	best_iteration = 0
	net = BaselineCNN()
	net = net.cuda()

	loss_func = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate)

    # define loss function
	# define optimizer
	# for each iteration, iteratively train all batches
	for i in range(0, num_iter):
		print(f"Iteration {i}")

		train_loss = 0
		net.train()
		for point in train_loader:
			data, label = point
			data = data.cuda()
			label = label.cuda()
			pred_label = net(data)
			loss = loss_func(pred_label, label)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
		train_loss = train_loss / len(train_loader)
		torch.save(net.state_dict(), f"epoch{i}.pt")
		avg_train_loss.append(train_loss)


		te_loss = 0
		net.eval()
		for point in val_loader:
			data, label = point
			data = data.cuda()
			label = label.cuda()
			pred_label = net(data)
			loss = loss_func(pred_label, label)
			te_loss += loss.item()
		te_loss = te_loss / len(val_loader)
		avg_val_loss.append(te_loss)

		if te_loss < best_val_score:
			best_val_score = te_loss
			best_iteration = i

	net.load_state_dict(torch.load(f"epoch{best_iteration}.pt"))
	return net, avg_train_loss, avg_val_loss

"""
My modified Convolutional Neural Network (CNN) model, based on baseline

Initialize CNN model    --> define loss function 
						--> define optimizer
						--> train your model with num_iter epochs 
						--> pick the best model and return
    - parameters: 	train_loader --- the train dataloader
					val_loader --- the validation dataloader
	- Return:     	net --- the best trained MLP network with the lowest validation loss
                  	avg_train_loss --- a list of averaged training loss of length num_iter
                  	avg_val_loss --- a list of averaged validation loss of length num_iter

"""
def trainCustom(train_loader,val_loader):
    # average training loss, one value per iteration (averaged over all batches in one iteration)
    avg_train_loss = []

    # average validation loss, one value per iteration (averaged over all batches in one iteration)
    avg_val_loss = []

    # record the lowest validation loss, used to determine early stopping (best model)
    best_val_score = float('inf')
	best_iteration = 0
	net = CNNPlusPlus()
	net = net.cuda()

	loss_func = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

    # define loss function
	# define optimizer
	# for each iteration, iteratively train all batches
	for i in range(0, num_iter):
		print(f"Iteration {i}")

		train_loss = 0
		net.train()
		for point in train_loader:
			data, label = point
			data = data.cuda()
			label = label.cuda()
			pred_label = net(data)
			loss = loss_func(pred_label, label)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
		train_loss = train_loss / len(train_loader)
		torch.save(net.state_dict(), f"epoch{i}.pt")
		avg_train_loss.append(train_loss)


		te_loss = 0
		net.eval()
		for point in val_loader:
			data, label = point
			data = data.cuda()
			label = label.cuda()
			pred_label = net(data)
			loss = loss_func(pred_label, label)
			te_loss += loss.item()
		te_loss = te_loss / len(val_loader)
		avg_val_loss.append(te_loss)

		if te_loss < best_val_score:
			best_val_score = te_loss
			best_iteration = i

	net.load_state_dict(torch.load(f"epoch{best_iteration}.pt"))
	return net, avg_train_loss, avg_val_loss

"""
Evaluate the model, using unseen data features "X" and
corresponding labels "y".

	Parameters: loader --- the test loader
				net --- the best trained network
	Return: 	the accuracy on test set
"""
def evaluate(loader, net):
	total = 0
	correct = 0

	# use model to get predictions
	for X, y in loader:
		net = net.cuda()
		outputs = net(X)
		predictions = torch.argmax(outputs.data, 1)

		# total number of items in dataset
		total += y.shape[0]

		# number of correctly labeled items in dataset
		correct += torch.sum(predictions == y)

	# return fraction of correctly labeled items in dataset
	return float(correct) / float(total)

"""
Start everything here
"""
if __name__ == "__main__":

	use_mlp = False
	use_custom = True
	torch.cuda.init()

	# load data from file
	#train_loader, val_loader, test_loader = \
		#read_data('hw0train.txt','hw0validate.txt', 'hw0test.txt')
	train_loader, val_loader, test_loader = \
		read_data(train_data, val_data, test_data)

	if use_mlp:
		net, t_losses, v_losses = trainMLP(train_loader,val_loader)
	elif use_custom:
		net, t_losses, v_losses = trainCustom(train_loader,val_loader)
	else:
		net, t_losses, v_losses = trainCNN(train_loader,val_loader)

	# evaluate model on validation data
	# accuracy = evaluate(test_loader, net)
	# evaluate model on test data
	accuracy = evaluate(test_loader, net)

	print("Test accuracy: {}".format(accuracy))

	# plot losses
	plt.plot(t_losses)
	plt.plot(v_losses)
	plt.legend(["training_loss","validation_loss"])
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Loss plot")
	plt.show()