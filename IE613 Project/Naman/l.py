#Contextual Combinatorial Cascading Bandits

import numpy as np

import math

import matplotlib.pyplot as plt

import scipy.stats as ss

import csv

#get parameters
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
	
	


def deriv_divergence( p,q ): #not needed
	if p==0:
		if q==1:
			return math.inf
		return ((1-p)/(1-q))
	elif p==1:
		if q==0:
			return math.inf
		return -p/q
	else:
		return (((1-p)/(1-q))-(p/q))

	
    




# Function to find the root 
def kl_solver(mu_icap,counts, time): 
	err = 0.0001
	q_val = np.zeros(len(mu_icap),dtype = float)
	for i in range(len(mu_icap)):
		p = mu_icap[i]
		left = p
		right = 1
		if p==1:
			q_val[i] = 1
		else:
			while 1:
				mid = (left+right)/2
				kl = divergence(p,mid)
				rhs = (math.log(time+1)+3*math.log(math.log(time+3)))/counts[i]
				if abs(kl - rhs) < err:
					break
				if kl-rhs<0:
					left = mid
				else:
					right = mid
			q_val[i] = mid

	return q_val



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
	

def CascadeKLUCB(T,w):
	w1 = np.random.binomial(1,w)
	N = np.ones(L)
	U = np.zeros(L)
	w0 = w1
	regret = np.zeros(T)
	for t in range(T):
		# for i in range(L):
		# 		if(t==0):
		# 			U[i] = w1[i]

		# 		else: 
		# 			U[i] = w1[i] + math.sqrt((1.5*math.log(t))/N[i])
		U = kl_solver(w1,N,t )
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
				
			
		
		regret[t] = regret[t-1] + get_regret(w,At)
		print(t,regret[t],At)
	return regret


T=100000
L = 16
K = 2
a1 = 0.2
delta = 0.15
a2 = a1-delta
w = np.array([a1 if i<K else a2 for i in range(L)])



reg=0
#for i in range(20):
reg += CascadeKLUCB(100000,w)
#reg/=20
plt.plot(range(T),reg,'r',label='L=16,K=2')
plt.legend(loc='upper right', numpoints=1)
plt.title("cumulative regret")
plt.xlabel("t")
plt.ylabel("cumulative regret")

plt.show()

plt.close()