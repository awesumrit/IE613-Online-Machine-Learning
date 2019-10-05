#Contextual Combinatorial Cascading Bandits

import numpy as np

import math

import matplotlib.pyplot as plt

import scipy.stats as ss

import csv

#get parameters
def get_context():
	Xtp= np.random.rand(d-1)

	Xtp = Xtp/np.linalg.norm(Xtp)

	Xt_ap = np.zeros(d)

	Xt_ap[0:d-1] = Xtp

	Xt_ap[d-1] = 1
	return Xt_ap

def Oracle(U_t):
	a = np.argsort(U_t)[::-1][:K]
	
	return a 


L = 200

K = 4

gamma = 0.9

gamma_k = np.array([gamma**(k) for k in range(K)])

d = 20

theta = np.random.rand(d-1)

theta = theta/np.linalg.norm(theta)

theta_st = np.zeros(d)

theta_st[0:d-1] = theta/2

theta_st[d-1] = 1/2
#disunctive_objective


def get_regret(w,gamma_k,At):
	a = 0
	for i in range(K):
		prod = 1
		for b in range(i):
			prod*=1-w[At[b]]
		a = a +gamma_k[i]*prod*w[At[i]]
	a_pr =  Oracle(w)
	a_st = 0
	for i in range(K):
		prod=1
		for b in range(i):
			prod*=1-w[a_pr[b]]
		a_st += gamma_k[i]*prod*w[a_pr[i]]
	return a_st-a


def C3_UCB(gamma_k, lamda, T):
	delta = 1/math.sqrt(T)
	theta_cap = np.zeros(d)
	beta = 1
	Vt =lamda*np.identity(d)
	Xt_a = np.zeros((L,d))
	Xt = np.zeros((0,0))
	Yt = np.zeros(0)
	Ut_a = np.zeros(L)
	regret = 0
	cum_regret = np.zeros(T)

	for t in range(T):
		for i in range(L):
			Xt_a[i] = get_context()

		for i in range(L):
			X = np.dot(theta_cap.T,Xt_a[i]) + beta*(np.dot(np.dot(Xt_a[i],np.linalg.inv(Vt)),Xt_a[i])) 

			Ut_a[i] = min(1,X)
		At = Oracle(Ut_a)
		means = np.dot(np.array([Xt_a[At[0]],Xt_a[At[1]],Xt_a[At[2]],Xt_a[At[3]]]),theta_st)
		Ot = 0
		#print(means,theta_st.dot(Xt_a[At[0]]))
		wt_a = np.random.binomial(1,means)
		for i in range(K):
			if wt_a[i]==1:
				Ot = i
				break
		
		print(Ot,end=' ')
		temp1 = np.array([gamma_k[i]*Xt_a[At[i]] for i in range(Ot+1)])
		#print(temp1.shape,Xt.shape)
		temp2 = np.array([gamma_k[i]*wt_a[i] for i in range(Ot+1)])
		Xt = temp1 if Xt.shape==(0,0) else np.append(Xt,temp1,axis=0)
		Yt = np.append(Yt,temp2)
		Vt = lamda*np.identity(d) + Xt.T.dot(Xt)
		#print(Xt.shape)
		theta_cap = np.linalg.inv(Vt).dot(Xt.T).dot(Yt)
		R=0.5
		beta = R*math.sqrt(math.log(np.linalg.det(Vt)/((lamda**d)*(delta**2))))+math.sqrt(lamda)
		
		regret = regret + get_regret(np.dot(Xt_a,theta_st),gamma_k,At)
		cum_regret[t] = regret
		print(t,regret)
	return cum_regret


lamda = 100
T = 5000
regret = C3_UCB(gamma_k,lamda,T)
plt.plot(range(T),regret,'r',label='disjunctive,gamma=0.9')
plt.legend(loc='upper right', numpoints=1)
plt.title("cumulative regret")
plt.xlabel("t")
plt.ylabel("cumulative regret")
plt.ylim([0,50.0])
plt.show()

plt.close()
