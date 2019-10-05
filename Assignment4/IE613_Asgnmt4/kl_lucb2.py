import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import scipy.stats as ss

p=0.5

k = 20

num_sample_paths = 20
numArms = k
num_arms = k 

alpha = 2
delta = 0.1
epsilon = 0
def select_arm(arm):
	if arm == 1:
		return np.random.binomial(1, p)
	else :
		return np.random.binomial(1, float(70*p-float(arm))/70)

def beta(t,K):
	k1 = 4*math.exp(1)+4
	return (math.log(k1*K*(t**alpha)/delta))+math.log(math.log(k1*K*(t**alpha)/delta))

def divergence(p,q):
	if p==0:
		if q==1:
			return math.inf
		else:
			return (1-p)*math.log((1-p)/(1-q))

	elif p==1:
		if q==0:
			return math.inf
		else:
			return p*math.log(p/q)
	else:
		return (p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q)))

def updateQMax_upper(mu_hat, counts, time):
  #eMeanUpdate(reward, pArmPulled);
	delta = 1e-2
	epsilon_er = 1e-4
	qMax = np.zeros(len(mu_hat),dtype = float)
	for i in range(len(mu_hat)):
		p = mu_hat[i]
		prev = p
		end = 1
		if p==1:
			qMax[i] = 1
		elif p==0:
			qMax[i] = 1-math.exp(-(beta(time,K)/counts[i]))
		else:
			while 1:
				mid = (prev+end)/2
				kl = divergence(p,mid)
				rhs = beta(time,K)/counts[i]
				if abs(kl - rhs) < epsilon_er:
					break
				if kl-rhs<0:
					prev = mid
				else:
					end = mid
			qMax[i] = mid
	return qMax

def updateQMax_lower(mu_hat, counts, time):
  #eMeanUpdate(reward, pArmPulled);
	delta = 1e-2
	epsilon_er = 1e-4
	qMax = np.zeros(len(mu_hat),dtype = float)
	for i in range(len(mu_hat)):	
		p = mu_hat[i]
		prev = 0
		end = p
		if p==1:
			qMax[i] = 1-math.exp(-(beta(time,K)/counts[i]))
		elif p==0:
			qMax[i] = 0
		else:
			while 1:
				mid = (prev+end)/2
				kl = divergence(p,mid)
				rhs = beta(time,K)/counts[i]
				if abs(kl - rhs) < epsilon_er:
					break
				if kl-rhs<0:
					end = mid
				else:
					prev = mid
			qMax[i] = mid
	return qMax


def kl_L_UCB(num_arms):
	m = int(K/5)
	k = num_arms
	Upper_confidence = np.zeros(K)
	Lower_confidence = np.zeros(K)
	qMax = np.zeros(k) 
	counts = np.zeros(k)
	values = np.zeros(k)
	loss = np.zeros(k)
	x = np.zeros(k)
	nx = np.ones(k)
	mu_hat = np.zeros(k)
	numTotalPulls = 0
	cumulativeReward = 0

	max_mu = p
	B = float('Inf')
	time_complex = 1
	counts = np.ones(K)
	Xt= np.zeros(K)
	for explore in range(k):
		It = explore
		reward = select_arm(It)
		values[It] += reward
		loss[It] += 1-reward
		counts[It] +=1
		cumulativeReward += reward

	mu_hat = values/counts
	
	Upper_confidence = updateQMax_upper(mu_hat, counts, time_complex)
	
	Lower_confidence = updateQMax_lower(mu_hat, counts, time_complex)
	lower_t=0
	upper_t=0
	Jt = np.zeros(m)
	Jt = mu_hat.argsort()[-m:][::-1]
	for i in range(m):
		bound = float('Inf')
		if bound > Lower_confidence[Jt[i]]:
			bound = Lower_confidence[Jt[i]]
			lower_t = Jt[i]
		else:
			pass
	bound = 0
	Jc = np.argpartition(mu_hat, K-m)
	Jc = Jc[:K-m]
	for i in range(K-m):
		if bound < Upper_confidence[Jc[i]]:
			bound = Upper_confidence[Jc[i]]
			upper_t = Jc[i]
		else:
			pass

	while B > epsilon:
		Jt = mu_hat.argsort()[-m:][::-1]
		values[upper_t] += select_arm(upper_t)
		values[lower_t] += select_arm(lower_t)
		counts[upper_t] += 1
		counts[lower_t] += 1 
		mu_hat = values/counts
		
		
		time_complex += 1
		Upper_confidence = updateQMax_upper(mu_hat, counts, time_complex)
		Lower_confidence = updateQMax_lower(mu_hat, counts, time_complex)
		for i in range(m):
			bound = float('Inf')
			if bound > Lower_confidence[Jt[i]]:
				bound = Lower_confidence[Jt[i]]
				lower_t = Jt[i]
			else:
				pass
		bound = 0
		Jc = np.argpartition(mu_hat, K-m)
		Jc = Jc[:K-m]
		for i in range(K-m):
			if bound < Upper_confidence[Jc[i]]:
				bound = Upper_confidence[Jc[i]]
				upper_t = Jc[i]
			else:
				pass
		


		B = Upper_confidence[upper_t] - Lower_confidence[lower_t]
	
	return time_complex, Jt



average_sample_complexity = np.zeros(5)
mistake_probability = np.zeros(5)
K = 10
num_instances = 1
Total_time = 0
mistake = 0
m = int(K/5)
for j in range(5):

	mistake = 0
	Total_time = 0
	for it in range(num_instances):
		#print(it)
		
		time , Jt= kl_L_UCB(K)
		Total_time += time

		a1 = np.sort(Jt)
		a2 = mu.argsort()[-m:][::-1]
		mistake += np.array_equal(a1, a2)

		

	average_sample_complexity[j] = Total_time/num_instances
	mistake_probability[j] = mistake/num_instances
	K +=10

print average_sample_complexity
print mistake_probability

