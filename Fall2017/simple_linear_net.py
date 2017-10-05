import torch 
import numpy as np 
import torch.nn as nn 
import torch.optim as optim 
from torch.autograd import Variable
import matplotlib.pyplot as plt 


"""
This script generates and classifies a spiral dataset using a two layer 
fully connected neural network in PyTorch.
"""
N = 100 # Number of elements per class
D = 2 # Dimensionality of the data
K = 3 # number of classes
def gen_spiral_dataset(N=100, D=2, K=3):
	X = np.zeros((N*K, D))
	y = np.zeros(N*K, dtype=np.uint8)
	for j in range(K):
		ind = range(N*j, N*(j+1))
		r = np.linspace(0,0.1,N)
		t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2
		X[ind] = np.c_[r*np.sin(t), r*np.cos(t)]
		y[ind] = j 
	return X, y


class Net(nn.Module):
	"""
	Net: A simple three layer neural network. 
	"""
	def __init__(self):
		super(Net, self).__init__()
		self.layer1 = nn.Linear(D, 64) # Linear layer: y = Wx + b 
		self.layer2 = nn.Linear(64, K)
		self.softmax = nn.LogSoftmax()

	def forward(self, x):
		x1 = self.layer1(x)
		x2 = self.layer2(x1)
		x3 = self.softmax(x2)
		return x3


net = Net() 
print(net)

X,y = gen_spiral_dataset()
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

lossfn = nn.NLLLoss()
optimz = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
var_y = Variable(torch.from_numpy(y).type(torch.LongTensor))
var_x = Variable(torch.from_numpy(X).type(torch.FloatTensor), requires_grad=True)
	
def train(net):
	net.train()
	
	for ep in range(30):
		ov_loss = 0.0
		for i in range(N*K):
			optimz.zero_grad()
			op = net(var_x)
			loss = lossfn(op, var_y)
			loss.backward()
			optimz.step()
			
			if i%100 == 0:
				print("Epoch:%d, Iteration:%d,  Loss: %.3f"%(ep, i, loss.data[0]))
		
	print("Finished training\n")


def eval(net):
	net.eval()
	op = net(var_x[0, :])
	max_val = op.data.max(0)[1]
	
	


if __name__ == '__main__':
	train(net)
	eval(net)

