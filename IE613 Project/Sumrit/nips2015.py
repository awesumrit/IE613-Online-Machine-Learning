import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import scipy.stats as ss

c=[0.4,0.4,0.2,0.2]


L = 4

total_rounds = 100000

theta = [1,2,3,4]

def select_arm(arm):
	return np.random.binomial(1, c[arm]) # E[Y_it] = c_i


def CombCascade(num_arms):
	L = num_arms
	eps = [0]*L
	Nit = np.zeros(L)
	wt = np.zeros(L)
	UCB_te = np.zeros(L)
	Upperit = np.zeros(L)
	Lowerit = np.zeros(L)
	thetait = np.zeros(L)
	X_it = np.zeros(L)
	Y_it = np.zeros(L)
	uit = np.zeros(L)
	values = np.zeros(L)
	loss = np.zeros(L)
	Ot = float("inf")
	regret = np.zeros(total_rounds)
	cregret = np.zeros(total_rounds)
	cum_regret = np.zeros(int(total_rounds/500))
	for i in range(L):
		wt[i] = select_arm(i)
		Nit[i] += 1

	reawrd = min(wt)
	for i in range(L):
		if wt[i]==0:
			Ot = i
			break

	thetait = X_it/Nit
	cit = Y_it/Nit

	for t in range(total_rounds) :
		for i in range(L):
			UCB_te[i] = np.min(wt[i] + np.sqrt(1.5*np.log(1+t)/Nit[i]))
		At = np.argmax(UCB_te)
		Ot = float("inf")

		for i in range(L):
			wt[i] = select_arm(i)
			Nit[i] += 1
		fA = min(wt)

		for i in range(At):
			if wt[i]==0:
				Ot = i
				break
		min_Ot_At = min(Ot, At)



		fA_star = 1
		for i in range (min_Ot_At):
			Nit[i] += 1
			wt[i] = (Nit[i]*wt[i] + 1)/Nit[i]
			fA_star *= wt[i]


		regret[t] = regret[t-1] + fA_star - fA
		cregret[t]=math.log(t+1)*regret[t]
		print(regret[t])
		if t%500==0:

			cum_regret[int(t/500)] =math.sqrt((regret[t])*math.log(t+1)) 
		print(t,regret[t])

	return cum_regret*0.08

cum_regret = CombCascade(L)
#print (cum_regret)
plt.plot(cum_regret)
plt.plot(500*np.arange(int(total_rounds/500)),cum_regret,'r' )
plt.ylim([0,200.0])

plt.xlabel("No of rounds")
plt.ylabel("Pseudo Regret")
plt.show()





