#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import gym


# In[3]:


eps=1e-10
DTYPE = torch.float32


# In[288]:


class Net(nn.Module):
    def __init__(self, state_size, action_size, is_continuous=False):
        super().__init__()
        self.is_continuous = is_continuous
        
        pi_model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
        value_model = nn.Sequential(
            nn.Linear(state_size, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        
        self.pi = pi_model
        self.ve = value_model
        
    def forward(self, x):
        pass
    
    def getValue(self, x):
        x = torch.tensor(x, dtype=DTYPE)
        return self.ve(x)
    
    def getPi(self, x):
        x = torch.tensor(x, dtype=DTYPE)
        probs = self.pi(x)
        return nn.Softmax(dim=-1)(probs) if not self.is_continuous else probs 
    
    def setContinuous(self):
        self.is_continuous = True


# In[289]:


class ReplayBuffer():
    def __init__(self, minl, maxl,alpha,sample_size=400):
        self.max_length = maxl
        self.length = 0
        self.buffer = []
        self.weights = []
        self.alpha = alpha
        self.min_length = minl
        self.sample_size = sample_size
        
    def add(self, ep):
        
        weight = np.exp(self.alpha*np.array(ep['rewards']).sum())+eps
#         if self.length>0 and weight < 0.75*np.min(self.weights) + 0.25 * np.mean(self.weights) and self.trainable(): return

        self.buffer.append(ep)
        self.length += 1

        self.weights.append(weight)
        
        if self.length > self.max_length:
            idx = np.random.randint(self.length)
            del self.buffer[idx]
            del self.weights[idx]
            self.length-=1
            
    def sample(self):
        wts = np.array(self.weights)
        
        sample_size = min(self.sample_size, self.length)
        sample = np.random.choice(self.buffer,size=(sample_size,), p=wts/wts.sum(), replace=False)
        out = sample if self.length>1 else [sample]
        return out
    
    def trainable(self):
        return self.length>self.min_length
    


# In[314]:


class PCL(object):
    def __init__(self, env, replay_buffer, state_size, action_size, epoch=1000, off_policy_rate=20, pi_lr=7e-4, 
                 ve_lr=3.5e-4, entropy_tau=0.15, rollout_d=20, gamma=1):
        self.epoch = epoch
        self.state_size =  state_size
        self.action_size = action_size
        self.net = Net(self.state_size, self.action_size)
        self.net.is_continuous = False
        
        self.replay_buffer = replay_buffer
        self.env = env
        
        self.pi_optimiser = optim.Adam(self.net.pi.parameters(), lr=pi_lr)
        self.ve_optimiser = optim.Adam(self.net.ve.parameters(), lr=ve_lr)
        self.off_policy_rate = off_policy_rate
        self.tau = entropy_tau
        self.rollout_d = rollout_d
        self.gamma = gamma
    
    def selectProb(self, pi, a):
        a = np.eye(self.action_size, dtype=np.int32)[a]
        pi_selected = torch.sum(pi * torch.tensor(a, dtype=DTYPE), dim=1)
        return torch.log(pi_selected+eps)

        
    def optimise(self, episodes):
        net = self.net
        
        self.pi_optimiser.zero_grad()
        self.ve_optimiser.zero_grad()
        
#         print("before loop", episodes)
        for episode in episodes:
            rollout_d = min(self.rollout_d, len(episode['states']))
            ma = len(episode['states'])-rollout_d+1
            
            for i in range(ma):
                states = episode['states'][i:i+rollout_d]
                a = episode['actions'][i:i+rollout_d]  
                R = episode['rewards'][i:i+rollout_d]

                ve_init = net.getValue(states[0])
                ve_end = net.getValue(states[-1])

                pi_all_states = net.getPi(states).squeeze(0)
                log_pi_selected = self.selectProb(pi_all_states, a)

                discount = torch.tensor(self.gamma**np.arange(0,rollout_d))

                c = -ve_init + self.gamma**rollout_d*ve_end                + torch.sum(discount*(torch.tensor(R) - self.tau*log_pi_selected))

                loss = c**2
                loss.backward()
                
#                 print("----------------------------", loss)
#         print("after loop")
        
        self.ve_optimiser.step()
        self.pi_optimiser.step()
        
    def getStateEnc(self,x):
        return x
    def getStateDec(self,x):
        return x
    def getActionEnc(self,a):
        return a
    def getActionDec(self,a):
        return a
    
    def getAction(self, state):
        enc_state = self.getStateEnc(state)
        
        pi = self.net.getPi(enc_state).squeeze(0).detach().numpy()
        return self.getActionDec(np.random.choice(np.arange(self.action_size), p=pi))
    
    def rollout(self, max_ep_length = -1):
        states = []
        actions = []
        rewards = []
        env = self.env
        
        state = env.reset()
        is_terminated = False
        timestep = 0
        
        while not is_terminated and timestep!=max_ep_length:
            action = self.getAction(state)
            next_state, reward, is_terminated, _ = env.step(action)
            
            states.append(self.getStateEnc(state))
            actions.append(self.getActionEnc(action))
            rewards.append(reward)
            
            state = next_state
            timestep += 1

        
        rewards = np.array(rewards)
#         print(np.mean(rewards))
#         rewards = (rewards - np.mean(rewards))
        
#         print(len(states))
#         print("")
        return dict(
            states = states,
            actions = actions,
            rewards = list(rewards)
        )
    
    def train(self, max_ep_length=-1):
        tot_rewards = []
        for i in range(self.epoch):
            episode = self.rollout(max_ep_length)        
            self.optimise([episode])
            
            r = np.array(episode['rewards'])
            tot_rewards.append(r.sum())
            
            if i>20: 
                print("I: {},    B:{}       R: {}                    ".format(i, self.replay_buffer.length, np.mean(tot_rewards[-20:])), end='\r')
            else:
                print("I: {},    B:{}       R: {}                    ".format(i, self.replay_buffer.length, tot_rewards[-1]), end='\r')
                
            self.replay_buffer.add(episode)
            
            ########off policy
            if self.replay_buffer.trainable():
                for _ in range(self.off_policy_rate):
                    episodes = self.replay_buffer.sample()
                    self.optimise(episodes)
            
#             print(i)
#             for x in self.net.pi.parameters():
#                 print(x)


# In[315]:


class ContinuousPCL(PCL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net.setContinuous()
        
    def selectProb(self, params, a):
        """
        params: [mean, var]
        """
        
        mean = params[:, 0]
        std = params[:, 1]
        a = torch.tensor(a, dtype=DTYPE)
        
        return -(a-mean)**2/(std**2+eps) - std  
    
    def getAction(self, state):
        enc_state = self.getStateEnc(state)
        params = self.net.getPi(enc_state).squeeze(0).detach()
        action = Normal(params[0], params[1]).sample().numpy()
        return np.array([action])
    
    def getActionEnc(self, action):
        return torch.tensor(action, dtype=DTYPE)


# In[316]:


class CopyPCL(PCL):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def getStateEnc(self,x):
        return np.array([x])
    
    def getStateDec(self,x):
        return x[0]
    
    def getActionEnc(self,a):
        return a[0]*6 +(1-a[1])*5+ a[2]
    
    def getActionDec(self,a):
        return (a//6, 1-(a%6)//5, (a%6)%5)
    


# In[317]:


env = gym.make('Copy-v0').env
buffer = ReplayBuffer(32, 10000, 1)
agent = CopyPCL(env, buffer, 1, 12, epoch=100000, off_policy_rate=20,entropy_tau=0.05, pi_lr=0.005, ve_lr=0.0025)
# agent.getStateEnc = lambda x: np.eye(6)[int(x)]
# agent.getStateDec = lambda x: np.argmax(x)


# In[ ]:


agent.train(100)


# In[245]:


agent.getAction(0)


# In[ ]:





# In[ ]:




