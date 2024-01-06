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
        self.yt_normalization = Normalization(shape=1)
        self.ut_normalization = Normalization(shape=1)
        self.yt_target_normalization = Normalization(shape=1)

    def step(self, action, phi_1, phi_2, current_timestep, max_timesteps_per_episode) :
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
            self.delta_array =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            self.nt_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            self.yt_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            self.yt_m_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            self.phi_array_1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            self.phi_array_2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            # self.delta_array.append(delta)
            self.ut_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            # self.rew_array.append(0)
            self.yt_target_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            self.yt_target_m_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            # self.ut_normalarray = [0,0,0]
            # self.yt_normalarray = [0,0,0]
            # self.yt_target_normalarray = [0,0,0]
            self.done = False
            self.current_timestep_array = [0,0,0]


        # get yt (sensitive analysis verision)
        ut = action
        delta = np.random.uniform(low=-1, high=1)
        nt = self.nt_array[-1] + delta - self.delta_array[-1]
        yt = phi_1 * self.yt_array[-1] + phi_2 * self.yt_array[-2] + 2 + 2 * ut - 1 * current_timestep + nt
        yt_target = abs(yt - self.target)

        # if yt_target > 1e+20 : 
        #     yt_target = 1e+20
        # if yt > 1e+20 : 
        #     yt = 1e+20
        # yt_m = yt / 10000
        # yt_target_m = yt_target / 10000


        self.ut_array.append(float(ut))
        self.nt_array.append(float(nt))
        self.yt_array.append(float(yt))
        self.yt_target_array.append(float(yt_target))  
        self.yt_target_m_array.append(float(yt_target))
        self.yt_m_array.append(float(yt))
        self.phi_array_1.append(phi_1)
        self.phi_array_2.append(phi_2)
        self.delta_array.append(delta)
        self.current_timestep_array.append(current_timestep)

        # create obs (phi, yt, ut, |yt-target|)
        seq1 = np.array((self.phi_array_1[-1], self.phi_array_2[-1], self.nt_array[-1], self.yt_m_array[-1], self.ut_array[-1], self.yt_target_m_array[-1], self.delta_array[-1]))             # capture last 3 info to obs 
        seq2 = np.array((self.phi_array_1[-2], self.phi_array_2[-2], self.nt_array[-2], self.yt_m_array[-2], self.ut_array[-2], self.yt_target_m_array[-2], self.delta_array[-2]))   
        seq3 = np.array((self.phi_array_1[-3], self.phi_array_2[-3], self.nt_array[-3], self.yt_m_array[-3], self.ut_array[-3], self.yt_target_m_array[-3], self.delta_array[-3]))
        seq4 = np.array((self.phi_array_1[-4], self.phi_array_2[-4], self.nt_array[-4], self.yt_m_array[-4], self.ut_array[-4], self.yt_target_m_array[-4], self.delta_array[-4]))
        seq5 = np.array((self.phi_array_1[-5], self.phi_array_2[-5], self.nt_array[-5], self.yt_m_array[-5], self.ut_array[-5], self.yt_target_m_array[-5], self.delta_array[-5]))
        seq6 = np.array((self.phi_array_1[-6], self.phi_array_2[-6], self.nt_array[-6], self.yt_m_array[-6], self.ut_array[-6], self.yt_target_m_array[-6], self.delta_array[-6]))
        seq7 = np.array((self.phi_array_1[-7], self.phi_array_2[-7], self.nt_array[-7], self.yt_m_array[-7], self.ut_array[-7], self.yt_target_m_array[-7], self.delta_array[-7]))
        seq8 = np.array((self.phi_array_1[-8], self.phi_array_2[-8], self.nt_array[-8], self.yt_m_array[-8], self.ut_array[-8], self.yt_target_m_array[-8], self.delta_array[-8])) 
        seq9 = np.array((self.phi_array_1[-9], self.phi_array_2[-9], self.nt_array[-9], self.yt_m_array[-9], self.ut_array[-9], self.yt_target_m_array[-9], self.delta_array[-9])) 
        seq10 = np.array((self.phi_array_1[-10], self.phi_array_2[-10], self.nt_array[-10], self.yt_m_array[-10], self.ut_array[-10], self.yt_target_m_array[-10], self.delta_array[-10]))   
        seq11 = np.array((self.phi_array_1[-11], self.phi_array_2[-11], self.nt_array[-11], self.yt_m_array[-11], self.ut_array[-11], self.yt_target_m_array[-11], self.delta_array[-11]))   
        seq12 = np.array((self.phi_array_1[-12], self.phi_array_2[-12], self.nt_array[-12], self.yt_m_array[-12], self.ut_array[-12], self.yt_target_m_array[-12], self.delta_array[-12]))   
        seq13 = np.array((self.phi_array_1[-13], self.phi_array_2[-13], self.nt_array[-13], self.yt_m_array[-13], self.ut_array[-13], self.yt_target_m_array[-13], self.delta_array[-13]))   
        seq14 = np.array((self.phi_array_1[-14], self.phi_array_2[-14], self.nt_array[-14], self.yt_m_array[-14], self.ut_array[-14], self.yt_target_m_array[-14], self.delta_array[-14]))   
        seq15 = np.array((self.phi_array_1[-15], self.phi_array_2[-15], self.nt_array[-15], self.yt_m_array[-15], self.ut_array[-15], self.yt_target_m_array[-15], self.delta_array[-15]))   
      
        obs = np.vstack((seq15, seq14, seq13, seq12, seq11, seq10, seq9, seq8, seq7, seq6, seq5, seq4, seq3, seq2, seq1))
      
        #reward 
        if abs(yt_target) <= 2: 
            rew = (-15 * yt_target) + 60
        elif abs(yt_target) <= 5: 
            rew = (-6.666 * yt_target) + 43.33
        elif abs(yt_target) <= (self.epsilon):
            rew = (-2 * yt_target) + 20 
        else :
            # rew = np.clip(-abs(yt - self.target), -1e+5, 1e+5)
            rew = -np.log2(abs(yt_target))
        self.rew_array.append(rew)

        # find the time of episode done 
        # if time_step >200 or the five times |yt-target| in epslion contiunously 
        num = 0 
        for i in self.rew_array[-6:]:
            if i == self.constant : 
                num += 1 

        if current_timestep >= max_timesteps_per_episode:
            self.done = True 

        return obs, rew, self.done, yt_target, yt

    def reset(self, phi_1, phi_2, max_time_per_episode): 
        
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
        obs ,_ ,_ ,_ ,_= self.step(0, phi_1, phi_2, max_timesteps_per_episode=max_time_per_episode,current_timestep=0)
        return obs

        


