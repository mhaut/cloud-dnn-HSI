import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class autoencoder(nn.Module):
	def __init__(self, topology):
		super(autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(topology[0], topology[1]),
			nn.ReLU(True),
			nn.Linear(topology[1], topology[2]),
			nn.ReLU(True)
		)
		self.decoder = nn.Sequential(
			nn.Linear(topology[2], topology[3]),
			nn.ReLU(True),
			nn.Linear(topology[3], topology[4])
		)
	def forward(self, x):
		x_enc = self.encoder(x)
		x_dec = self.decoder(x_enc)
		return x_dec



# globals VARIABLES need for LBFGS
step = 0
final_loss = None
iters = 400
########################



#ALERT: LOAD YOUR DATA HERE
#features_train = 

features_train = torch.from_numpy(features_train.astype('float32'))
input_shape = features_train.shape[1]

features_train = features_train.cuda()
data = Variable(features_train)


topology = [input_shape, 140, 60, 140, input_shape]

model = autoencoder(topology).cuda()
optimizer = optim.LBFGS(model.parameters(), max_iter=iters, history_size=100, lr=1.0)#, line_search_fn='Wolfe')


def closure():
	global step, final_loss
	optimizer.zero_grad()
	output = model(data)
	loss = F.mse_loss(output, data)
	print("Step %3d loss %6.5f"%(step, loss.data[0]))
	step+=1
	if step == iters:
		final_loss = loss.item()
	loss.backward()
	return loss


final_loss = optimizer.step(closure)

print("FINAL LOSS", final_loss.item())
