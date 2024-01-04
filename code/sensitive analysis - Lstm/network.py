import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np

class actor_FeedForwardNN(nn.Module):

	def __init__(self, input_dim, out_dim, hidden_size, hidden_size_2, num_layers, batch_size, output_scale_factor):
		
		super(actor_FeedForwardNN, self).__init__()

		self.hidden_size = hidden_size
		self.hidden_size_2 = hidden_size_2
		self.num_layers = num_layers
		self.batch_size = batch_size
		self.output_scale_factor = output_scale_factor

		# lstm
		# self.layer1 = nn.Linear(input_dim, hidden_size)
		# self.activation1 = nn.ReLU()
		self.layer2 = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=False, bidirectional=True)
		self.layer3 = nn.Linear(hidden_size * 2, 256)
		self.activation3 = nn.ReLU()
		self.layer4 = nn.Linear(256, 256)
		self.activation4 = nn.ReLU()
		self.layer5 = nn.Linear(256, 128)
		self.activation5 = nn.ReLU()
		self.layer6 = nn.Linear(128, 64)
		self.activation6 = nn.ReLU()
		self.layer7 = nn.Linear(64, 64)
		self.activation7 = nn.ReLU()
		self.layer8 = nn.Linear(64, 1)
		# self.layer6_1 = nn.Linear(64, 1)
		# self.activation6 = nn.Softplus()

		# self.fc_v = nn.Linear(128, 1)
		

	def forward(self, obs, hidden):
		"""
			Runs a forward pass on the neural network.
		"""
		
		# Convert observation to tensor if it's a numpy array 
		# if isinstance(obs, np.ndarray):
		# 	obs = torch.tensor(obs, dtype=torch.float)

		# out1 = self.layer1(obs)
		# out2 = self.activation1(out1)
		x, lstm_hidden = self.layer2(obs, hidden)

		x = self.layer3(x)
		x = self.activation3(x)

		x = self.layer4(x)
		x = self.activation4(x)

		x = self.layer5(x)
		x = self.activation5(x)

		x = self.layer6(x)
		x = self.activation6(x)

		x = self.layer7(x)
		x = self.activation7(x)

		x = self.layer8(x)
		output = (x[-1,:].squeeze())* self.output_scale_factor
		# output2 = self.activation6(out10_1)
		# output2 = (output2[-1,:].squeeze()) + 0.5  * self.var_scale_factor
		# output2 = (output2[-1,:].squeeze()) * 10

		return output, lstm_hidden
	
	def mult_forward(self, obs, hidden):
		# for batch_obs, batch_hidden 

		h0_list = [h_0[0].detach() for h_0 in hidden]
		h1_list = [h_1[1].detach() for h_1 in hidden]

		h0_tensor = torch.cat(h0_list, dim=1)
		h1_tensor = torch.cat(h1_list, dim=1)

		hidden_tuple = (h0_tensor, h1_tensor)
		
		# out1, lstm_hidden = self.layer2(obs, hidden_tuple)
		# out2 = self.layer3(out1)
		# out3 = self.activation3(out2)
		# out4 = self.layer4(out3)
		# out5 = self.activation4(out4)
		# out6 = self.layer5(out5)
		# out7 = self.activation5(out6)
		# out8 = self.layer6(out7)
		# output = out8[-1,:].squeeze()

		# out1 = self.layer1(obs)
		# out2 = self.activation1(out1)
		x, lstm_hidden = self.layer2(obs, hidden_tuple)

		x = self.layer3(x)
		x = self.activation3(x)

		x = self.layer4(x)
		x = self.activation4(x)

		x = self.layer5(x)
		x = self.activation5(x)

		x = self.layer6(x)
		x = self.activation6(x)

		x = self.layer7(x)
		x = self.activation7(x)

		x = self.layer8(x)
		output = (x[-1,:].squeeze()) * self.output_scale_factor
		# output2 = self.activation6(out10_1)
		# output2 = (output2[-1,:].squeeze()) + 0.5 * self.var_scale_factor
		# output2 = (output2[-1,:].squeeze())

		# for i in range(self.batch_size): 
		# 	x_i = obs[ :, i:i+1, :]
		# 	h_i = hidden[i]
		# 	out, lstm_hidden = self.layer2(x_i, h_i)
		# 	out = self.layer3(out)
		# 	out = self.activation2(out)
		# 	output = self.layer4(out)
		# 	V_batch.append(output[-1])
		# 	lstm_hidden_batch.append(lstm_hidden)

		return output, lstm_hidden
 
	
class critic_FeedForwardNN(nn.Module):

	def __init__(self, input_dim, out_dim, hidden_size, hidden_size_2, num_layers, batch_size, output_scale_factor):
		
		super(critic_FeedForwardNN, self).__init__()

		self.hidden_size = hidden_size
		self.hidden_size_2 = hidden_size_2
		self.num_layers = num_layers
		self.batch_size = batch_size
		self.output_scale_factor = output_scale_factor

		# lstm 
		self.layer2 = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=False, bidirectional=True)
		self.layer3 = nn.Linear(hidden_size * 2, 128)
		self.activation3 = nn.ReLU()
		self.layer4 = nn.Linear(128, 128)
		self.activation4 = nn.ReLU()
		self.layer5 = nn.Linear(128, 128)
		self.activation5 = nn.ReLU()
		self.layer6 = nn.Linear(128, 64)
		self.activation6 = nn.ReLU()
		self.layer7 = nn.Linear(64, 1)
		
		
		# self.layer3 = nn.Linear(input_dim, 512)
		# self.activation3 = nn.Tanh()
		# self.layer4 = nn.Linear(512, 512)
		# self.activation4 = nn.Tanh()
		# self.layer5 = nn.Linear(512, 512)
		# self.activation5 = nn.Tanh()
		# self.layer6 = nn.Linear(512, 256)
		# self.activation6 = nn.Tanh()
		# self.layer7 = nn.Linear(256, 256)
		# self.activation7 = nn.Tanh()
		# self.layer8 = nn.Linear(256, 128)
		# self.activation8 = nn.Tanh()
		# self.layer9 = nn.Linear(128, 128)
		# self.activation9 = nn.Tanh()
		# self.layer10 = nn.Linear(128, 64)
		# self.activation10 = nn.Tanh()
		# self.layer11 = nn.Linear(64, 64)
		# self.activation11 = nn.Tanh()
		# self.layer12 = nn.Linear(64, 1)

	def forward(self, obs, hidden):
		"""
			Runs a forward pass on the neural network.
		"""
		
		# Convert observation to tensor if it's a num`p`y array 
		# if isinstance(obs, np.ndarray):
		# 	obs = torch.tensor(obs, dtype=torch.float)

		# hidden_0 = []
		# hidden_1 = []
		# for i in hidden: 
		# 	hidden_0.append(i[0])
		# 	hidden_1.append(i[1])	dw
		# hidden_0 = tuple(hidden_0)
		# hidden_1 = tuple(hidden_1)

		h0_list = [h_0[0].detach() for h_0 in hidden]
		h1_list = [h_1[1].detach() for h_1 in hidden]

		h0_tensor = torch.cat(h0_list, dim=1)
		h1_tensor = torch.cat(h1_list, dim=1)

		hidden_tuple = (h0_tensor, h1_tensor)

		x, lstm_hidden = self.layer2(obs, hidden_tuple)

		x = self.layer3(x)
		x = self.activation3(x)

		x = self.layer4(x)
		x = self.activation4(x)

		x = self.layer5(x)
		x = self.activation5(x)

		x = self.layer6(x)
		x = self.activation6(x)

		x = self.layer7(x)
		# x = self.activation7(x)

		# x = self.layer8(x)
		# x = self.activation8(x)

		# x = self.layer9(x)
		# x = self.activation9(x)

		# x = self.layer10(x)
		# x = self.activation10(x)

		# x = self.layer11(x)
		# x = self.activation11(x)

		# x = self.layer12(x)
		output = x[-1, :].squeeze()
				
		# V_batch = []
		# lstm_hidden_batch = []
 
		# for i in range(self.batch_size):
		# 	x_i = obs[:, i:i+1, :]
		# 	h_i = hidden[i]
		# 	out, lstm_hidden = self.layer2(x_i, h_i)
		# 	out = self.layer3(out)
		# 	out = self.activation2(out)
		# 	output = self.layer4(out)
		# 	V_batch.append(output[-1])
		# 	lstm_hidden_batch.append(lstm_hidden)


		return output, lstm_hidden