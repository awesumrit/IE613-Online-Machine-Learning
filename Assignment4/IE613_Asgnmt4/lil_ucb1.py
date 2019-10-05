import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import scipy.stats as ss

p=0.5
T = 25000
k = 20
alpha = 1.5
num_sample_paths = 20
numArms = k
num_arms = k 
epsilon = 0.01
beta = 1
lmbda = 9
sigma = 0.5

def select_arm(arm):
	if arm == 1:
		return np.random.binomial(1, p)
	else :
		return np.random.binomial(1, float(70*p-float(arm))/70)



def lil_ucb(mu, num_arms):
	k = num_arms

	counts = np.zeros(k)
	values = np.zeros(k)
	loss = np.zeros(k)
	x = np.zeros(k)
	nx = np.ones(k)

	max_mu = p
	mu_hat = np.zeros(k)
	numTotalPulls = 0
	cumulativeReward = 0
	for explore in range(k):
		It = explore
		reward = np.random.binomial(size=1, n=1, p= mu[explore])
		values[It] += reward
		loss[It] += 1-reward
		counts[It] +=1
		cumulativeReward += reward

	time = k

	mu_hat = values/counts

	index = np.argmax(counts)

	while(counts[index]<(1+lmbda*np.sum(counts) - lmbda*counts[index])) :
		Aj = np.argmax(mu_hat + (1+beta)*(1+np.sqrt(epsilon))* np.sqrt(2*(sigma**2)*(1+epsilon)*np.log(np.log(1+epsilon)*counts)/counts) )
		reward = np.random.binomial(size=1, n=1, p= mu[Aj])
		values[Aj] += reward
		loss[Aj] += 1-reward
		counts[Aj] += 1
		mu_hat = values/counts
		time += 1

		cumulativeReward += reward
		index = np.argmax(counts)

	return time, np.argmax(counts)


average_sample_complexity = np.zeros(5)
mistake_probability = np.zeros(5)
K = 10
num_instances = 10
Total_time = 0
mistake = 0
m = int(K/5)
for j in range(5):
	mu = np.random.uniform(low=0.0, high=1.0, size=K)
	mistake = 0
	Total_time = 0
	for it in range(num_instances):
		#print(it)
		
		time , best_pred = lil_ucb(mu,K)
		Total_time += time

		mistake += (best_pred==np.argmax(mu))
		

	average_sample_complexity[j] = Total_time/num_instances
	mistake_probability[j] = mistake/num_instances
	K +=10

print average_sample_complexity
print mistake_probability
