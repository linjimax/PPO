import time

import numpy as np
import time
import torch
import math
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist 
import math 
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from tensorboardX import SummaryWriter
from normalization import  Normalization, RewardScaling


class PPO:

	def __init__(self, policy_class, critic_class, env, tensorboard_write, excel_sheet, **hyperparameters):
		"""
			policy_class is network of actor and critic
		"""
	
		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)
		torch.set_printoptions(profile="full")

		# GPU 
		self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
	
		# Extract environment information 
		self.env = env
		self.obs_dim = 6                               # depend on s
		self.batch_obs_dim = 12
		self.act_dim = 1                                # controller ut 

		 # Initialize actor and critic networks
		self.actor = policy_class(self.obs_dim, self.act_dim, hidden_size=self.hidden_size, hidden_size_2=self.hidden_size_2, num_layers=self.num_layers, batch_size=self.timesteps_per_batch, output_scale_factor=self.output_scale_factor)                                                   # ALG STEP 1
		self.critic = critic_class(self.obs_dim, self.act_dim, hidden_size=self.hidden_size, hidden_size_2=self.hidden_size_2, num_layers=self.num_layers, batch_size=self.timesteps_per_batch, output_scale_factor=self.output_scale_factor)  
		# self.actor.to(self.device)
		# self.critic.to(self.device)


		# for name, param in self.actor.named_parameters():
		# 	print(f'{name}: {param.data.shape}')	

		# for name, param in self.actor.named_parameters():
		# 	print(f'{name}: {param.data}')	
		# 	print('************************************')
		
		# # he initialization for actor and critic network 
		# layer1_node = self.actor.layer1.out_features
		# layer3_node = self.actor.layer3.out_features
		# layer4_node = self.actor.layer4.out_features
		# layer5_node = self.actor.layer5.out_features
		# layer6_node = self.actor.layer6.out_features
		# layer7_node = self.actor.layer7.out_features
		# layer8_node = self.critic.layer8.out_features
		# layer9_node = self.critic.layer9.out_features
		# layer10_node = self.critic.layer10.out_features

		# layer7_node = self.actor.layer7.out_features
		# layer8_node = self.actor.layer8.out_features
		# layer_dims_vector_actor = [self.hidden_size, layer3_node, layer4_node, layer5_node, layer6_node, layer7_node,  self.act_dim]
		# layer_dims_vector_critic = [self.obs_dim, layer3_node, layer4_node, layer5_node, layer6_node, layer7_node, layer8_node, layer9_node, layer10_node, self.act_dim]
		# params_dist2 = self.initialize_parameters_he(layer_dims_vector_actor)
		# params_dist3 = self.initialize_parameters_he(layer_dims_vector_critic)
		# with torch.no_grad():
		# 	# self.actor.layer1.weight = nn.Parameter(torch.tensor(params_dist2['Weight1'], dtype=torch.float))
			# self.actor.layer3.weight = nn.Parameter(torch.tensor(params_dist2['Weight1'], dtype=torch.float))
			# self.actor.layer4.weight = nn.Parameter(torch.tensor(params_dist2['Weight2'], dtype=torch.float))
			# self.actor.layer5.weight = nn.Parameter(torch.tensor(params_dist2['Weight3'], dtype=torch.float))
			# self.actor.layer6.weight = nn.Parameter(torch.tensor(params_dist2['Weight4'], dtype=torch.float))
			# self.actor.layer7.weight = nn.Parameter(torch.tensor(params_dist2['Weight5'], dtype=torch.float))
		# 	self.actor.layer8.weight = nn.Parameter(torch.tensor(params_dist2['Weight6'], dtype=torch.float))

		# 	# self.actor.layer1.bias = nn.Parameter(torch.tensor(params_dist2['bias1'], dtype=torch.float).squeeze())
			# self.actor.layer3.bias = nn.Parameter(torch.tensor(params_dist2['bias1'], dtype=torch.float).squeeze())
			# self.actor.layer4.bias = nn.Parameter(torch.tensor(params_dist2['bias2'], dtype=torch.float).squeeze())
			# self.actor.layer5.bias = nn.Parameter(torch.tensor(params_dist2['bias3'], dtype=torch.float).squeeze())
			# self.actor.layer6.bias = nn.Parameter(torch.tensor(params_dist2['bias4'], dtype=torch.float).squeeze())
			# self.actor.layer7.bias = nn.Parameter(torch.tensor(params_dist2['bias5'], dtype=torch.float).squeeze())
		# 	self.actor.layer8.bias = nn.Parameter(torch.tensor(params_dist2['bias6'], dtype=torch.float))

			# self.critic.layer3.weight = nn.Parameter(torch.tensor(params_dist3['Weight1'], dtype=torch.float))
			# self.critic.layer4.weight = nn.Parameter(torch.tensor(params_dist3['Weight2'], dtype=torch.float))
			# self.critic.layer5.weight = nn.Parameter(torch.tensor(params_dist3['Weight3'], dtype=torch.float))
			# self.critic.layer6.weight = nn.Parameter(torch.tensor(params_dist3['Weight4'], dtype=torch.float))
			# self.critic.layer7.weight = nn.Parameter(torch.tensor(params_dist3['Weight5'], dtype=torch.float))
			# self.critic.layer8.weight = nn.Parameter(torch.tensor(params_dist3['Weight6'], dtype=torch.float))
			# self.critic.layer9.weight = nn.Parameter(torch.tensor(params_dist3['Weight7'], dtype=torch.float))
			# self.critic.layer10.weight = nn.Parameter(torch.tensor(params_dist3['Weight8'], dtype=torch.float))

			# self.critic.layer3.bias = nn.Parameter(torch.tensor(params_dist3['bias1'], dtype=torch.float).squeeze())
			# self.critic.layer4.bias = nn.Parameter(torch.tensor(params_dist3['bias2'], dtype=torch.float).squeeze())
			# self.critic.layer5.bias = nn.Parameter(torch.tensor(params_dist3['bias3'], dtype=torch.float).squeeze())
			# self.critic.layer6.bias = nn.Parameter(torch.tensor(params_dist3['bias4'], dtype=torch.float).squeeze())
			# self.critic.layer7.bias = nn.Parameter(torch.tensor(params_dist3['bias5'], dtype=torch.float).squeeze())
			# self.critic.layer8.bias = nn.Parameter(torch.tensor(params_dist3['bias6'], dtype=torch.float).squeeze())
			# self.critic.layer9.bias = nn.Parameter(torch.tensor(params_dist3['bias7'], dtype=torch.float).squeeze())
			# self.critic.layer10.bias = nn.Parameter(torch.tensor(params_dist3['bias8'], dtype=torch.float))

		self.actor.to(self.device)
		self.critic.to(self.device)

		# xavier_normal initialization 
		self.actor.apply(self.initialize_parameters_xavier)
		self.critic.apply(self.initialize_parameters_xavier)

		# orthogonal initialization 
		# nn.init.orthogonal_(self.actor.layer1.weight)
		# nn.init.orthogonal_(self.actor.layer3.weight)
		# nn.init.orthogonal_(self.actor.layer4.weight)
		# nn.init.orthogonal_(self.actor.layer5.weight)
		# nn.init.orthogonal_(self.actor.layer6.weight)
		# nn.init.orthogonal_(self.actor.layer7.weight)
		# nn.init.orthogonal_(self.actor.layer8.weight)
		#nn.init.orthogonal_(self.actor.layer9.weight)
		# nn.init.constant_(self.actor.layer1.bias, 0.5)
		nn.init.constant_(self.actor.layer3.bias, 0)
		nn.init.constant_(self.actor.layer4.bias, 0)
		nn.init.constant_(self.actor.layer5.bias, 0)
		nn.init.constant_(self.actor.layer6.bias, 0)
		nn.init.constant_(self.actor.layer7.bias, 0)
		nn.init.constant_(self.actor.layer8.bias, 0)
		# nn.init.constant_(self.actor.layer9.bias, 0)


		# nn.init.orthogonal_(self.critic.layer1.weight)
		# nn.init.orthogonal_(self.critic.layer3.weight)
		# nn.init.orthogonal_(self.critic.layer4.weight)
		# nn.init.orthogonal_(self.critic.layer5.weight)
		# nn.init.constant_(self.critic.layer1.bias, 0)
		nn.init.constant_(self.critic.layer3.bias, 0)
		nn.init.constant_(self.critic.layer4.bias, 0)
		nn.init.constant_(self.critic.layer5.bias, 0)
		nn.init.constant_(self.critic.layer6.bias, 0)
		nn.init.constant_(self.critic.layer7.bias, 0)
		# nn.init.constant_(self.critic.layer8.bias, 0)
		# nn.init.constant_(self.critic.layer9.bias, 0)
		# nn.init.constant_(self.critic.layer10.bias, 0)
		# nn.init.constant_(self.critic.layer11.bias, 0)
		# nn.init.constant_(self.critic.layer12.bias, 0)




		# for layer in range(1,2):
		# 	getattr(self.actor, 'weight').data = torch.tensor(params_dist['Weight'+str(layer)])
		# 	self.actor[layer].bias.data = torch.tensor(params_dist['bias'+str(layer+1)])

		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters() ,lr=self.lr )
		self.critic_optim = Adam(self.critic.parameters() ,lr=self.lr)

		# learning rate decay 
		# self.scheduler_actor = torch.optim.lr_scheduler.ReduceLROnPlateau(self.actor_optim, mode='max', factor=0.5, patience=15, verbose=True, threshold=0.003, threshold_mode='abs')

		# tensorboard info 
		self.write = tensorboard_write
		self.total_step = 0

		#excel info 
		self.excel_sheet = excel_sheet

		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
			'critic_losses': [],
			'STD_ut': []
		}

	def learn(self, total_timesteps):
		"""
			Parameters:
			total_timesteps - the total number of timesteps to train for
		"""
		
		print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		while t_so_far < total_timesteps:                                                                       # ALG STEP 2

			# for name, param in self.actor.named_parameters():
			# 	print(f'{name}: {param.data}')	

			# print('**********************************/n')

			# for name, param in self.critic.named_parameters():
			# 	print(f'{name}: {param.data}')	
			#print("更新前的 acitva1 alpha 參數值:", self.actor.activation1.weight.data)
			#print("更新前的 acitva2 alpha 參數值:", self.actor.activation2.weight.data)
			#print("更新前的 acitva3 alpha 參數值:", self.actor.activation3.weight.data)
		#print("更新前的 acitva4 alpha 參數值:", self.actor.activation4.weight.data)
			#%print("更新前的 acitva5 alpha 參數值:", self.actor.activation5.weight.data)
			#print("更新前的 acitva6 alpha 參數值:", self.actor.activation6.weight.data)


			# Initialize the covariance matrix used to query the actor for actions
			if i_so_far >= 310 :
				self.cov_var = torch.full(size=(self.act_dim,), fill_value=50.0, device=self.device)
			elif i_so_far >= 260 :
				self.cov_var = torch.full(size=(self.act_dim,), fill_value=100.0, device=self.device)
			elif i_so_far >= 210 :
				self.cov_var = torch.full(size=(self.act_dim,), fill_value=150.0, device=self.device)
			elif i_so_far >= 150 :
				self.cov_var = torch.full(size=(self.act_dim,), fill_value=250.0, device=self.device)
			else : 
				self.cov_var = torch.full(size=(self.act_dim,), fill_value=500.0, device=self.device)
			self.cov_mat = torch.diag(self.cov_var).to(device=self.device)

			batch_obs, batch_acts, batch_log_probs, batch_lens, batch_rews, batch_masks, first_hidden, second_hidden = self.rollout()                     # ALG STEP 3

			# first_hidden = torch.stack(first_hidden, dim=1).squeeze()
			# second_hidden = torch.stack(second_hidden, dim=1).squeeze()

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# Logging timesteps so far and iterations so far
			self.logger['t_so_far'] = t_so_far
			self.logger['i_so_far'] = i_so_far

			# Calculate advantage at k-th iteration
			# V, V_prime,_ = self.evaluate(batch_obs, batch_acts, first_hidden, second_hidden)
			V, V_prime ,_ = self.evaluate(batch_obs, batch_acts, first_hidden, second_hidden)
			A_k, delta_v= self.compute_gae(V, V_prime, batch_masks, batch_rews, self.gamma, self.gae_lmbda)					# ALG STEP 5
			# A_k, delta_v= self.compute_gae(V, batch_masks, batch_rews, self.gamma, self.gae_lmbda )
			# batch_rtgs = self.compute_rtgs(batch_episode_rews)                                   

			# Normalizing advantages
			# but in practice it decreases the variance of our advantages and makes convergence much more stable and faster.
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			batch_log_probs = torch.stack(batch_log_probs, dim=0)

			# initial critic_lstm hidden state 
			# hidden_init_batch = []
			# hidden_init = (torch.ones([self.num_layers, 1, self.hidden_size_2], dtype=torch.float, device=self.device),
			# 	  			torch.ones([self.num_layers, 1, self.hidden_size_2], dtype=torch.float, device=self.device))
			# for i in range(self.timesteps_per_batch):  
			# 	hidden_init_batch.append(hidden_init)

			torch.cuda.empty_cache()

			# This is the loop where we update our network for some n epochs
			for j in range(self.n_updates_per_iteration):                                                     # ALG STEP 6 & 7
				
				for i in range(40):
					
					# Calculate V_phi and pi_theta(a_t | s_t)
					V, _, curr_log_probs= self.evaluate(batch_obs, batch_acts, first_hidden, second_hidden)
					critic_loss = nn.MSELoss()(V, delta_v)

					print(f'critic_loss[ {i} ] = {critic_loss}')

					# initial critic_lstm hidden state 
					# _, _, _ = self.evaluate(batch_obs, batch_acts, hidden_init_batch, hidden_init_batch)

					# for name, param in self.actor.named_parameters():
							# print(f'{name}: {param.grad}')	

					# Calculate gradients and perform backward propagation for critic network
					self.critic_optim.zero_grad()
					critic_loss.backward(retain_graph=True)

					# for name, param in self.critic.named_parameters():
					# 	if param.grad is not None:
					# 		self.writer.add_histogram("gradients/" + name, param.grad, i)

					# gradient clip 
					# if self.use_grad_clip: 
					# 	torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
					self.critic_optim.step()

					# for name, param in self.critic.named_parameters():
					# 	if param.grad is not None:
					# 		gradient_norm = torch.norm(param.grad)
					# 		print(f'Critic Layer: {name}, Gradient Norm: {gradient_norm}')

					self.logger['critic_losses'].append(critic_loss.detach().to(device='cpu'))

					torch.cuda.empty_cache()

				# Calculate V_phi and pi_theta(a_t | s_t)
				# _, _, curr_log_probs = self.evaluate(batch_obs, batch_acts, first_hidden, second_hidden)

					# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				ratios = torch.exp(curr_log_probs - batch_log_probs)
				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# # initial actor_lstm hidden state 
				# _, _, _ = self.evaluate(batch_obs, batch_acts, hidden_init_batch, hidden_init_batch)

				# Calculate actor and critic losses.
				actor_loss = (-torch.min(surr1, surr2)).mean()
				# critic_loss = nn.MSELoss()(V, delta_v)

				# for name, param in self.actor.named_parameters():
				# 	if param.grad is not None:
				# 		gradient_norm = torch.norm(param.grad)
				# 		print(f'Actor Layer: {name}, Gradient Norm: {gradient_norm}')
				# 		if gradient_norm > 1:
    			# 		# 梯度爆炸处理代码\
				# 			print('******************梯度爆炸****************')

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)

				# for name, param in self.actor.named_parameters():
				# 		if param.grad is not None:
				# 			self.writer.add_histogram("gradients/" + name, param.grad, i)
				

				# gradient clip 
				# if self.use_grad_clip: 
				# 	torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
				self.actor_optim.step()

				# # Calculate gradients and perform backward propagation for critic network
				# self.critic_optim.zero_grad()
				# critic_loss.backward()
				# self.critic_optim.step()

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach().to(device='cpu'))

				torch.cuda.empty_cache()

			
			# print("更新後的 acitva1 alpha 參數值:", self.actor.activation1.weight.data)
			# print("更新後的 acitva2 alpha 參數值:", self.actor.activation2.weight.data)
			# print("更新後的 acitva3 alpha 參數值:", self.actor.activation3.weight.data)
			# print("更新後的 acitva4 alpha 參數值:", self.actor.activation4.weight.data)
			# print("更新後的 acitva5 alpha 參數值:", self.actor.activation5.weight.data)
			# print("更新後的 acitva6 alpha 參數值:", self.actor.activation6.weight.data)

			
			#learning rate decay
			ave_rew = torch.mean(batch_rews)
			# self.scheduler_actor.step(ave_rew)
			new_actor_lr = self.actor_optim.param_groups[0]['lr']
			self.write.add_scalar('actor_lr', float(new_actor_lr), i_so_far)
				

			# Print a summary of our training so far
			self._log_summary()

			# Save our model if it's time
			if i_so_far % self.save_freq == 0:
				torch.save(self.actor.state_dict(), './ppo_actor.pth')
				torch.save(self.critic.state_dict(), './ppo_critic.pth')

			torch.cuda.empty_cache()


	def rollout(self):
		"""tensorboard --logdir=
            on-ploicy current , need to change to off-policy 
			Parameters:
				None

			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		# Batch data. For more details, check function header.
		batch_obs = []
		batch_acts = []
		batch_acts_np = []
		batch_log_probs = []
		batch_rews = []
		np_batch_rews = []
		batch_lens = []
		batch_masks = []
		masks = []
		phi_batch = []
		h_in_batch = []
		h_out_batch = []

		np.random.seed()
		random.seed()

		# # phi shuffle 
		# num_episode = math.ceil(self.timesteps_per_batch / 200)
		# if isinstance(num_episode/8, int) : 
		# 	for i in np.arange(num_episode/8) : 
		# 		phi_batch.append(0.2)
		# 		phi_batch.append(0.3)
		# 		phi_batch.append(0.4)
		# 		phi_batch.append(0.5)
		# 		phi_batch.append(0.6)
		# 		phi_batch.append(0.7)
		# 		phi_batch.append(0.8)
		# 		phi_batch.append(0.9)


		# else : 
		# 	for i in np.arange(int(num_episode/8)) : 
		# 		phi_batch.append(0.2)
		# 		phi_batch.append(0.3)
		# 		phi_batch.append(0.4)
		# 		phi_batch.append(0.5)
		# 		phi_batch.append(0.6)
		# 		phi_batch.append(0.7)
		# 		phi_batch.append(0.8)
		# 		phi_batch.append(0.9)
		# 	reminder = num_episode % 8 
		# 	for k in np.arange(reminder) : 
		# 		phi = round(np.random.uniform(low=0.2, high=0.9), 1)
		# 		phi_batch.append(phi)

		# np.random.shuffle(phi_batch)

		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
		ep_rews = []

		t = 0 # Keeps track of how many timesteps we've run so far this batch
		num_ep = 0

		# Keep simulating until we've run more than or equal to specified timesteps per batch
		while t < self.timesteps_per_batch:
			ep_rews = [] # rewards collected per episode
			masks = []

			# adjust phi(autocorrlation)
			phi = round(np.random.uniform(low=0.2, high=0.99), 1)
			# phi = phi_batch[num_ep]
			# num_ep += 1 

			# Reset the environment. sNote that obs is short for observation. 
			obs = self.env.reset(phi, self.max_timesteps_per_episode)
			done = False


			h_out = (torch.zeros([self.num_layers * 2, 1, self.hidden_size], dtype=torch.float, device=self.device), 
					torch.zeros([self.num_layers * 2, 1, self.hidden_size], dtype=torch.float, device=self.device))
			
			# h_out = torch.zeros([self.num_layers, 1, self.hidden_size_2], dtype=torch.float, device=self.device).detach()

			# normalization obs & reward_scaling initialization 
			# State_Normalization = Normalization(shape=(1,self.obs_dim))
			# Reward_scaling = RewardScaling(gamma=0.99, shape=1)

			# Run an episode for a maximum of max_timesteps_per_episode timesteps
			for ep_t in np.arange(1, self.max_timesteps_per_episode + 1 ):

				h_in = h_out

				t += 1 # Increment timesteps ran this batch so far (include episode timestep)

				# convert obs size to fit the neural network 
				obs = obs.reshape(8, 1, self.obs_dim)


				# Normalization obs 
				# obs = State_Normalization.__call__(obs)
				obs = torch.from_numpy(obs).to(device=self.device, dtype=torch.float32)

				# if ep_t == 1 : 
				# 	self.write.add_graph(self.actor, input_to_model = (torch.tensor(obs.clone().detach(), dtype=torch.float, device=self.device), h_in))

				# Track observations in this batch
				batch_obs.append(obs)

				# Calculate action and make a step in the env. 
				# rew in this episode 
				action, log_prob, h_out = self.get_action(obs, h_in)
				obs, rew, done, yt_target, _ = self.env.step(action, phi, ep_t, self.max_timesteps_per_episode)    
				mask = done
				mask = not mask

				# reward scaling 
				# scaling_rew = Reward_scaling.__call__(rew)


				# Track recent reward, action, and action log probability
				ep_rews.append(torch.tensor(rew))
				batch_acts.append(torch.tensor(action))
				batch_acts_np.append(action)
				batch_log_probs.append(log_prob.mean())
				masks.append(mask)
				h_in_batch.append(h_in)
				h_out_batch.append(h_out)

				# tensorboard 
				self.total_step += 1
				self.write.add_scalar('ut', float(action), self.total_step)
				self.write.add_scalar('rew', float(rew), self.total_step)
				self.write.add_scalar('phi', float(phi), self.total_step)
				self.write.add_scalar('yt_target', float(yt_target), self.total_step)
				self.write.flush()

				# If the environment tells us the episode is terminated, break
				if done == True : 
					break

			# Track episodic lengths and rewards
			batch_lens.append(ep_t)
			# batch_masks.append(masks)

			for i in ep_rews:
				batch_rews.append(i)
				np_batch_rews.append(i.item())
			for i in masks :
				batch_masks.append(i*1)

		std_ut = np.std(batch_acts_np)

		# Reshape data as tensors in the shape specified in function description, before returning
		batch_obs = [i.detach().clone().to(device=self.device) for i in batch_obs]
		batch_acts = [i.detach().clone().to(device=self.device) for i in batch_acts]
		batch_log_probs = [i.detach().clone().to(device=self.device) for i in batch_log_probs]
		# batch_rews = [torch.tensor(i, dtype=torch.float, device=self.device).view(1) for i in batch_rews]
		batch_rews = [i.detach().clone().to(device=self.device).view(1) for i in batch_rews]
		batch_rews = torch.stack(batch_rews).view(self.timesteps_per_batch)
		# batch_rews = [torch.tensor(i, dtype=torch.float) for i in batch_rews]

		# Log the episodic returns and episodic lengths in this batch.
		self.logger['batch_rews'] = np_batch_rews
		self.logger['batch_lens'] = batch_lens
		self.logger['STD_ut'] = std_ut

		torch.cuda.empty_cache()

		return batch_obs, batch_acts, batch_log_probs, batch_lens, batch_rews, batch_masks, h_in_batch, h_out_batch

	def compute_rtgs(self, batch_episode_rews):
		"""
			Compute the Reward-To-Go of each timestep in a batch given the rewards. 
			State-value-function
			Use MC control 
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
		batch_rtgs = []

		# Iterate through each episode
		for ep_rews in reversed(batch_episode_rews):

			discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
			# insert to 0 position 
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

		return batch_rtgs 
	
	def compute_gae (self, v, v_prime, batch_masks, batch_rewards, gamma, lmbda): 
		"""
			GAE 
			input : 
				v = state_value_function 
				masks = envriment done 
				rewards = 
				gamma = discount factor
				lmbda = smoothing factor
		"""
		batch_gae = []
		batch_delta_v = []
		gae = 0
		batch_rewards = batch_rewards.clone().detach()
		batch_masks = torch.tensor(batch_masks)

		for i in reversed(range(len(batch_rewards))):
			if i == (len(batch_rewards) - 1) :
				v_0 = torch.zeros(1).to(device=self.device)
				v = torch.cat((v, v_0), dim=0)
			delta = batch_rewards[i] + gamma * v_prime[i] * batch_masks[i] - v[i]
			gae = delta + gamma * lmbda * batch_masks[i] * gae
			batch_gae.insert(0, gae + v[i])
			delta_v = delta + v[i]
			batch_delta_v.insert(0, delta_v)
		
		batch_gae = torch.tensor(batch_gae, dtype=torch.float, device=self.device)
		batch_delta_v = torch.tensor(batch_delta_v, dtype=torch.float, device=self.device)
		
		return batch_gae, batch_delta_v
	
	def compute_gae_cnotc (self, v, batch_masks, batch_rewards, gamma, lmbda): 
		"""
			GAE 
			input : 
				v = state_value_function 
				masks = envriment done 
				rewards = 
				gamma = discount factor
				lmbda = smoothing factor
		"""
		batch_gae = []
		batch_delta_v = []
		gae = 0
		batch_rewards = batch_rewards.clone().detach()
		batch_masks = torch.tensor(batch_masks)

		for i in reversed(range(len(batch_rewards))):
			if i == (len(batch_rewards) - 1) :
				v_0 = torch.zeros(1).to(device=self.device)
				v = torch.cat((v, v_0), dim=0)
			delta = batch_rewards[i] + gamma * v[i-1] * batch_masks[i] - v[i]
			gae = delta + gamma * lmbda * batch_masks[i] * gae
			batch_gae.insert(0, gae + v[i])
			delta_v = delta + v[i]
			batch_delta_v.insert(0, delta_v)
		
		batch_gae = torch.tensor(batch_gae, dtype=torch.float, device=self.device)
		batch_delta_v = torch.tensor(batch_delta_v, dtype=torch.float, device=self.device)
		
		return batch_gae, batch_delta_v


	def get_action(self, obs, h_in):
		"""
			Queries an action from the actor network, should be called from rollout.
		"""

		# Query the actor network for a mean action
		mean, lstm_hidden = self.actor.forward(obs, h_in)
		mean = torch.tensor(mean.item(), device=self.device).view(1)
		# var = torch.tensor(var.item(), device=self.device).view(1)

		# std_noise_dist = dist.Normal(0,20)
		# std_noise = std_noise_dist.sample()
		# std = std.item() + abs(std_noise)
		# std = torch.tensor(std, device=self.device).reshape(1)

		normal_dist = dist.MultivariateNormal(mean, self.cov_mat)
		# normal_dist = dist.Normal(mean, y)

		# Sample an action from the distribution
		action = normal_dist.sample()

		# Calculate the log probability for that action
		log_prob = normal_dist.log_prob(action)

		action = action.item()
		# Return the sampled action and the log probability of that action in our distribution
		return action, log_prob.detach(), lstm_hidden
	
	


	def evaluate(self, batch_obs, batch_acts, first_hidden, second_hidden):
		"""
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.
		"""
		batch_obs = torch.stack(batch_obs, dim=1).squeeze()				# convert np shape to tensor 
		batch_acts = torch.stack(batch_acts, dim=0)							# same above 
	

		# reshape batch_obs to (num_timestep_pre_batch, batch_obs_shape[0]*[1])
		# the size that to fit with batch_rtg 
		# batch_obs_v = batch_obs.view(batch_obs.size(0), -1)
		batch_acts_v = batch_acts.view(batch_acts.size(0), -1)

		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
		# v_prime is t + 1 state, v is t state
		V_prime, _ = self.critic(batch_obs, second_hidden)
		torch.cuda.empty_cache()
		V, _ = self.critic(batch_obs, first_hidden)
		# V = self.critic(batch_obs)
		torch.cuda.empty_cache()

		# adjust V_prime, V size 
		# for i in V_prime : 
		# 	each_V_prime = i.item()			
		# 	V_prime_batch.append(each_V_prime)
		# V_prime_batch = torch.tensor(V_prime_batch, device=self.device, requires_grad=True)

		# for i in V : 
		# 	each_V = i.item()			
		# 	V_batch.append(each_V)
		# V_batch = torch.tensor(V_batch, device=self.device, requires_grad=True)


		# Value function normalization 
		# V = (V - V.mean()) / (V.std() + 1e-10)
		# V_prime = (V_prime - V_prime.mean()) / (V_prime.std() + 1e-10)

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
		mean, _ = self.actor.mult_forward(batch_obs, first_hidden)
		mean = mean.view(batch_obs.size(1), -1)
		# var = var.view(batch_obs.size(1), -1)

		# adjust mean size 
		# for i in mean : 
		# 	each_mean = i.item()			
		# 	mean_batch.append([each_mean])
		# mean_batch = torch.tensor(mean_batch, device=self.device, requires_grad=True)	
		
		normal_dist = dist.MultivariateNormal(mean, self.cov_mat)
		# normal_dist = dist.Normal(mean, y)
		log_probs = normal_dist.log_prob(batch_acts_v)		

		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
		torch.cuda.empty_cache()
		return V, V_prime ,log_probs

	def initialize_parameters_he(self, layer_dims) : 

		"""
    	initialize weights and biad as default value.
    	Arguments:
    	layer_dimes = a list contains nodes# in each layer including input features.
    	Returns:
    	parameters = a dictionary contains weights and bias
    	"""

		np.random.seed(1) #lock random parameters
		parameters = {}
		layers = len(layer_dims) 

		# count from first hidden layer
		for layer in range(1,layers): 
			if layer == 1:
				parameters["Weight"+str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer-1])
			else :
				parameters["Weight"+str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer-1])*np.sqrt(2/layer_dims[layer-1]) # He initialization
			parameters["bias"+str(layer)] = np.zeros((layer_dims[layer],1))
        
			assert(parameters['Weight' + str(layer)].shape == (layer_dims[layer], layer_dims[layer-1]))
			assert(parameters['bias' + str(layer)].shape == (layer_dims[layer],1))
                                              
		return parameters
	
	def initialize_parameters_xavier(self, model) :
		if isinstance(model, nn.Linear):
			torch.nn.init.xavier_uniform(model.weight)
			model.bias.data.fill_(0.01)

	def orthogonal_init(self, layer, gain=1.0):
		nn.init.orthogonal_(layer.weight, gain=gain)
		nn.init.constant_(layer.bias, 0)

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters
		"""
		
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch (total per batch)
		self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode 
		self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
		self.lr = 0.005                                 # Learning rate of actor optimizer
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
		self.gae_lmbda = 0.1		
		self.huber_delta = 20000
		self.use_grad_clip = True		
		self.hidden_size = 256
		self.hidden_size_2 = 128 
		self.num_layers = 1
		self.output_scale_factor=100
		self.var_scale_factor = 30

		# Miscellaneous parameters
		self.save_freq = 1                             # How often we save in number of iterations
		self.seed = None                                # Sets the seed of our program, used for reproducibility of results

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

	def _log_summary(self):
		"""
			Print to stdout what we've logged so far in the most recent batch.
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		delta_t = self.logger['delta_t']
		self.logger['delta_t'] = time.time_ns()
		delta_t = (self.logger['delta_t'] - delta_t) / 1e9
		delta_t = str(round(delta_t, 2))

		t_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_ep_lens = np.mean(self.logger['batch_lens'])
		avg_ep_rews = np.mean(self.logger['batch_rews'])
		avg_actor_loss = np.mean(self.logger['actor_losses'])
		avg_critic_loss = np.mean(self.logger['critic_losses'])
		STD_ut = self.logger['STD_ut']

		# tensorboard loss 
		self.write.add_scalar('actor_Loss', float(avg_actor_loss), i_so_far)
		self.write.add_scalar('critic_Loss', float(avg_critic_loss), i_so_far)
		self.write.add_scalar('average episodic reward', float(avg_ep_rews), i_so_far)
		self.write.add_scalar('STD_ut', float(STD_ut), i_so_far)

		# execel write 
		self.excel_sheet.cell(i_so_far + 1,1).value = avg_ep_rews 

		# Round decimal places for more aesthetic logging messages
		avg_ep_lens = str(round(avg_ep_lens, 2))
		avg_ep_rews = str(round(avg_ep_rews, 10))
		avg_actor_loss = str(round(avg_actor_loss, 10))

		#    logging statements
		print(flush=True)
		print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
		print(f"Average Episodic reward : {avg_ep_rews}", flush=True)
		print(f"Average Loss: {avg_actor_loss}", flush=True)
		print(f"STD_ut : {STD_ut}")
		print(f"Timesteps So Far: {t_so_far}", flush=True)
		print(f"Iteration took: {delta_t} secs", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []
		self.logger['critic_losses'] = []