#Thompson sampling in Cascading Bandits

import numpy as np

import math

import matplotlib.pyplot as plt

import scipy.stats as ss

import csv

L = 64
K = 2
a1 = 0.2
delta = 0.075
a2 = a1 - delta
w = np.array([a1 if i<K else a2 for i in range(L)])


def get_regret(w,S):
	a = 0
	
	prod = 1
	for j in range(K):
		prod *= (1-w[S[j]])
	a += (1-prod) 

	prod = 1
	for i in range(K):
		prod *= (1-w[i])
	return 1-prod-a
	


def thompson_cascade(T):
	mui_cap = np.zeros(L)
	N_i = np.zeros(L)
	regret = 0
	cum_regret = np.zeros(T)
	for t in range(T):
		Zt = np.random.normal(0,1)
		vi_cap = mui_cap*(1-mui_cap)
		sigma_i = np.maximum(np.sqrt(vi_cap*math.log(t+1)/(N_i+1)),(math.log(t+1))/(N_i+1))
		theta_i = mui_cap + Zt*sigma_i
		S_t = np.argsort(theta_i)[::-1][:K]
		Wt_i = np.random.binomial(1,w)
		for i in range(K):
			mui_cap[S_t[i]] = (N_i[S_t[i]]*mui_cap[S_t[i]]+Wt_i[S_t[i]])/(N_i[S_t[i]]+1) 
			N_i[S_t[i]] = N_i[S_t[i]]+1

		regret +=get_regret(w,S_t)
		cum_regret[t] = regret
		print(t,regret)
	return cum_regret

regret_ = 0
T = 100000
for i in range(20):
	regret_ += thompson_cascade(T)

regret_ /=20
plt.plot(range(T),regret_,'r',label='L=64delta=0.075')
plt.legend(loc='upper right', numpoints=1)
plt.title("cumulative regret")
plt.xlabel("t")
plt.ylabel("cumulative regret")
plt.ylim([0,7000.0])
plt.show()

plt.close()
