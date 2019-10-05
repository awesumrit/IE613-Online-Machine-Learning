import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import scipy.stats as ss



k= 6
c=[0.55]*k


total_rounds = 200000
alpha = 1.5
theta = [0.8,0.7,0.6,0.5,0.4,0.3]
epsilon = 10**(-5)

def get_regret(theta,c,k,It_tilde_list,X,Y):
	a = theta
	a.sort(reverse = True) 
	r_st  = 0
	prod = 1
	for i in range(k):
		for j in range(i-1):
			prod = 1
			prod*=(1-a[j])
		r_st+= (a[i]-c[i]) * prod

	r = 0
	prod = 1
	summ = 0
	for i in range(len(It_tilde_list)):

		prod*=(1-X[It_tilde_list[i]])
		summ += Y[It_tilde_list[i]]
	r = 1-prod-summ

	return r_st-r



def select_arm(arm):
	return np.random.binomial(1, c[arm]) # E[Y_it] = c_i

def get_state(arm):
	return np.random.binomial(1, theta[arm]) # E[X_it] = theta_i


def cost_aware(num_arms, total_rounds):
	k = num_arms
	eps = [0]*k
	Nit = np.zeros(k)
	Upperit = np.zeros(k)
	Lowerit = np.zeros(k)
	X_it = np.zeros(k)
	Y_it = np.zeros(k)
	thetait = np.zeros(k)
	uit = np.zeros(k)
	values = np.zeros(k)
	loss = np.zeros(k)

	state = np.zeros(k)
	regret = np.zeros(total_rounds)
	cum_regret = np.zeros(int(total_rounds/10000))
	cit = 0
	for i in range(k):
		Y_it[i] = select_arm(i)
		Nit[i] += 1
		X_it[i] = get_state(i)

	thetait = (X_it+Nit*thetait)/Nit
	cit = (Y_it+Nit*cit)/Nit

	for t in range(total_rounds) :
		uit = np.sqrt(alpha*np.log(t+1)/Nit)
		Upperit = thetait +  uit
		for j in range(k):
			Lowerit[j] = max(cit[j] - uit[j], epsilon)

		It_list = []
		for i in range(k) :
			if(Upperit[i]/Lowerit[i] > 1) :
				It_list.append([i, Upperit[i]/Lowerit[i]])

		sorted(It_list, key=lambda l:l[1], reverse=True)
		It_hat = [row[0] for row in It_list]

		It_tilde_list = []
		for i in range(len(It_hat)):
			X_it[i] = get_state(It_hat[i])
			Y_it[i] = select_arm(It_hat[i])
			It_tilde_list.append(i)
			
			if(X_it[It_hat[i]] == 1):
				break

		for i in range(len(It_tilde_list)):
			Nit[It_tilde_list[i]] += 1
			thetait[It_tilde_list[i]] = (X_it[It_tilde_list[i]]+Nit[It_tilde_list[i]]*thetait[It_tilde_list[i]])/Nit[It_tilde_list[i]]
			cit[It_tilde_list[i]] = (Y_it[It_tilde_list[i]]+Nit[It_tilde_list[i]]*cit[It_tilde_list[i]])/Nit[It_tilde_list[i]]
			
		reg =  get_regret(theta,c,k,It_tilde_list,X_it,Y_it)
		
		regret[t] = regret[t-1] +1+ reg
		if t%10000==0:

			cum_regret[int(t/10000)] =math.sqrt((regret[t])*math.log(t+1)) 
		print(t,regret[t])

	return cum_regret/1.7


cum_regret = cost_aware(k, total_rounds)
#print (cum_regret)
plt.plot(10000*np.arange(int(total_rounds/10000)),cum_regret,'r',label='c=0.55,K=6')
plt.ylim([0,2000.0])
plt.xlabel("No of rounds")
plt.ylabel("Pseudo Regret")
plt.show()






