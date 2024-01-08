import numpy as np 
import torch
from normalization import Normalization

class envri() :

    def __init__(self, target, reward_epsilon, reward_constant):
        """
            target --- process target 
            reward_epsilon --- |yt-target| <= epsilon 
            reward_constant  --- c constant in reward function 
        """
        self.target = target 
        self.epsilon = reward_epsilon
        self.constant = reward_constant
        self.delta_array = []
        self.phi_array = []
        self.nt_array = []
        self.yt_array = []
        self.ut_array = []
        self.yt_target = []
        self.rew_array = []
        self.obs = []
        self.done = False
        self.ut_normalarray = []
        self.yt_normalarray = []
        self.yt_target_normalarray = []
        # self.yt_normalization = Normalization(shape=1)
        # self.ut_normalization = Normalization(shape=1)
        # self.yt_target_normalization = Normalization(shape=1)

    def step(self, action, phi, current_timestep, max_timesteps_per_episode) :
        """
            get observation from control process 
            input: 
                action --- from actor
                current_time_step --- time_step in episode 
            output: 
                obs ---
                rew ---
        """
        if len(self.nt_array) == 0 and len(self.yt_array) == 0 :
            delta = np.random.uniform(low=-1, high=1)
            self.delta_array =[0,0,0,0,0,0,0]
            self.nt_array = [0,0,0,0,0,0,0,0]
            self.yt_array = [0,0,0,0,0,0,0,0]
            # self.phi_array = [0,0,0,0,0,0,0,0]
            self.delta_array.append(delta)
            self.ut_array = [0,0,0,0,0,0,0,0]
            self.rew_array.append(0)
            self.yt_target = [0,0,0,0,0,0,0,0]
            self.ut_normalarray = [0,0,0]
            self.yt_normalarray = [0,0,0]
            self.yt_target_normalarray = [0,0,0]
            self.done = False
            self.current_timestep_array = [0,0,0]

        # get yt (sensitive analysis verision)
        ut = action
        delta = np.random.uniform(low=-1, high=1)
        nt = self.nt_array[-1] + delta - self.delta_array[-1]
        yt = phi * self.yt_array[-1] + 2 + 2 * ut - 1 * current_timestep + nt
        yt_target = abs(yt - self.target)

        self.ut_array.append(float(ut))
        self.nt_array.append(float(nt))
        self.yt_array.append(float(yt))
        self.yt_target.append(float(yt_target))  
        # self.phi_array.append(phi)
        self.delta_array.append(delta)
        self.current_timestep_array.append(current_timestep)

        # create obs (phi, yt, ut, |yt-target|)
        seq1 = np.array((phi, self.nt_array[-1], self.yt_array[-1], self.ut_array[-1], self.yt_target[-1], self.delta_array[-1]))             # capture last 3 info to obs 
        seq2 = np.array((phi, self.nt_array[-2], self.yt_array[-2], self.ut_array[-2], self.yt_target[-2], self.delta_array[-2]))   
        seq3 = np.array((phi, self.nt_array[-3], self.yt_array[-3], self.ut_array[-3], self.yt_target[-3], self.delta_array[-3]))
        seq4 = np.array((phi, self.nt_array[-4], self.yt_array[-4], self.ut_array[-4], self.yt_target[-4], self.delta_array[-4]))
        seq5 = np.array((phi, self.nt_array[-5], self.yt_array[-5], self.ut_array[-5], self.yt_target[-5], self.delta_array[-5]))
        seq6 = np.array((phi, self.nt_array[-6], self.yt_array[-6], self.ut_array[-6], self.yt_target[-6], self.delta_array[-6]))
        seq7 = np.array((phi, self.nt_array[-7], self.yt_array[-7], self.ut_array[-7], self.yt_target[-7], self.delta_array[-7]))
        seq8 = np.array((phi, self.nt_array[-8], self.yt_array[-8], self.ut_array[-8], self.yt_target[-8], self.delta_array[-8]))   
        obs = np.vstack((seq8, seq7, seq6, seq5, seq4, seq3, seq2, seq1))
      
        #reward 
        if yt_target <= (self.epsilon) : 
            rew = (-33.33333 * yt_target) + 50 
        else :
            rew = -(yt_target)
        self.rew_array.append(rew)

        # find the time of episode done 0
        # if time_step >200 or the five times |yt-target| in epslion contiunously 
        num = 0 
        for i in self.rew_array[-6:] :
            if i == self.constant : 
                num += 1 

        if current_timestep >= max_timesteps_per_episode:
            self.done = True 

        return obs, rew, self.done, yt_target, yt

    def reset(self, phi, max_time_per_episode): 
        
        self.ut_array = []
        self.yt_array = []
        self.delta_array = []
        self.nt_array = []
        self.yt_target =[]
        self.rew_array =[]
        obs = []
        self.ut_normalarray = []
        self.yt_normalarray = []
        self.yt_target_normalarray = []
        # self.obs = torch.zeros((1,9),dtype=torch.float)
        obs ,_ ,_ ,_ ,_= self.step(0, phi, max_timesteps_per_episode=max_time_per_episode,current_timestep=0)
        return obs

        



