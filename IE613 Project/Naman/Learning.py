#Contextual Combinatorial Cascading Bandits

import numpy as np

import math

import matplotlib.pyplot as plt

import scipy.stats as ss

import csv

#get parameters


def Oracle(U_t):
	a = np.argsort(U_t)[::-1][:K]
	
	return a 




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
	

def CascadeUCB1(T,w):
	w1 = np.random.binomial(1,w)
	N = np.ones(L)
	U = np.zeros(L)
	w0 = w1
	regret = 0
	cum_regret = np.zeros(T)
	for t in range(T):
		for i in range(L):
				if(t==0):
					U[i] = w1[i]

				else: 
					U[i] = w1[i] + math.sqrt((1.5*math.log(t))/N[i])
		
		At = Oracle(U)
		
		Ot = 0
		w0 = np.random.binomial(1,w)
		for i in range(K):
			if w0[At[i]]==1:
				Ot = i
				break
			Ot = L+10
		

		for i in range(min(Ot+1,K)):
			N[At[i]] +=1
			w1[At[i]] = ((N[At[i]]-1)*w1[At[i]] + (Ot==i))/N[At[i]]
				
			
		
		regret = regret + get_regret(w,At)
		cum_regret[t] = math.sqrt(regret)*math.log(t+1)
		print(t,cum_regret[t],At)
	return 0.6*cum_regret


T=100000
L = 16
K = 4
a1 = 0.2
delta = 0.15
a2 = a1-delta
w = np.array([a1 if i<K else a2 for i in range(L)])

reg=0
#for i in range(20):
reg += CascadeUCB1(100000,w)
#reg/=20
plt.plot(range(T),reg,'r',label='L=16,K=2')
plt.legend(loc='upper right', numpoints=1)
plt.title("cumulative regret")
plt.xlabel("t")
plt.ylabel("cumulative regret")
plt.ylim([0,2000.0])
plt.show()

plt.close()