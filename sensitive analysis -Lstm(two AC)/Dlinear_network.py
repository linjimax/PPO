import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    
class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class Actor_FeedForwardNN_Dlinear_Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Actor_FeedForwardNN_Dlinear_Model, self).__init__()
        self.seq_len = configs["seq_len"]
        self.pred_len = configs["pred_len"]

        # Decompsition Kernel Size
        kernel_size = configs["kernel_size"]
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs["individual"]
        self.channels = configs["enc_in"]

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(device='cuda:1')
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(device='cuda:1')
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        x = x.permute(0,2,1) # to [Batch, Output length, Channel]

        return x[:,:,3] 
    


class Critic_FeedForwardNN_Dlinear_Model(nn.Module):

    def __init__(self, input_dim):
		
        super(Critic_FeedForwardNN_Dlinear_Model, self).__init__()

        # self.output_scale_factor = output_scale_factor

		# lstm 
        self.layer1 = nn.Linear(input_dim, 256)
		# self.activation1 = nn.ReLU()
        self.activation2 = nn.Tanh()
        self.layer3 = nn.Linear(256, 256)
        self.activation3 = nn.Tanh()
        self.layer4 = nn.Linear(256, 128)
        self.activation4 = nn.Tanh()
        self.layer5 = nn.Linear(128, 64)
        self.activation5 = nn.Tanh()
        self.layer6 = nn.Linear(64, 1)

    def forward(self, obs):
        """
			Runs a forward pass on the neural network.
		"""
        out3 = self.layer1(obs)
        out4 = self.layer3(out3)
        out5 = self.activation3(out4)
        out6 = self.layer4(out5)
        out7 = self.activation4(out6)
        out8 = self.layer5(out7)
        out9 = self.activation5(out8)
        out10 = self.layer6(out9)
        output = out10[:,-1,:].squeeze()
				
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


        return output

