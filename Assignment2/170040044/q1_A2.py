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
def select_arm(arm):
	if arm == 1:
		return np.random.binomial(1, p)
	else :
		return np.random.binomial(1, float(70*p-float(arm))/70)


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

def updateQMax(counts, rounds_until_now, mu_hat):
  #eMeanUpdate(reward, pArmPulled);
	delta = 1e-2;
	epsilon = 1e-4;
	qMax = np.zeros(numArms)
	for i in range(numArms):
		p = mu_hat[i]
		prev = p
		end = 1
		if p==1:
			qMax[i] = 1
		else:
			while 1:
				mid = (prev+end)/2
				kl = divergence(p,mid)
				rhs = np.log(rounds_until_now)/counts[i]
				if abs(kl - rhs) < epsilon:
					break
				if kl-rhs<0:
					prev = mid
				else:
					end = mid
			qMax[i] = mid
	return qMax

   
  
def e_greedy(total_rounds):

	eps = 0
	counts = np.ones(k)
	values = np.zeros(k)
	prob = np.zeros(k)
	for i in range(k):
		prob[i] = float(1)/k


	for explore in range(10*k):
		It = np.random.choice(np.arange(0,k), p=prob)
		values[It] += select_arm(It)
		counts[It] +=1
	
	mu_hat = np.zeros(k)

	regret = np.zeros(total_rounds)
	mu_hat = values/counts
	Aj = np.argmax(mu_hat)
	max_mu = p

	for t in range(10*k, total_rounds):
		eps = float(1)/t
		probs = np.array([1-eps if i == Aj else eps/(k-1) for i in range(k)])
		It = np.random.choice(np.arange(0,k), p=probs)
		reward = select_arm(It)
		# It
		values[It] += reward
		counts[It] += 1
		mu_hat = values/counts
		regret[t] = regret[t-1] + max_mu - (mu_hat[It]) 
		# mu_hat

	return regret

def kl_ucb(num_arms, total_rounds):
	k = num_arms
	eps = [0]*k
	qMax = np.zeros(k) 
	counts = np.zeros(k)
	values = np.zeros(k)
	loss = np.zeros(k)
	x = np.zeros(k)
	nx = np.ones(k)
	max_mu = p
	# min_mu
	cumulativeReward = 0
	mu_hat = np.zeros(k)

	regret = np.zeros(total_rounds)
	

	for explore in range(k):
		It = explore
		reward = select_arm(It)
		values[It] += reward
		loss[It] += 1-reward
		counts[It] +=1
		cumulativeReward += reward
		
	mu_hat = values/counts
	qMax = updateQMax(counts, k, mu_hat)

	for t in range(k, total_rounds):

		qMax = updateQMax(counts, t, mu_hat)
		Aj = np.argmax(qMax)
		reward = select_arm(Aj)
		values[Aj] += reward
		counts[Aj] += 1

		mu_hat = values/counts
		cumulativeReward += reward

		regret[t] += regret[t-1] + max_mu -(mu_hat[Aj])

	return regret

def ucb(num_arms, total_rounds):
	k = num_arms
	eps = [0]*k
	counts = np.zeros(k)
	values = np.zeros(k)
	loss = np.zeros(k)
	x = np.zeros(k)
	nx = np.ones(k)
	#min_mu  = 0

	max_mu = p
	mu_hat = np.zeros(k)
	numTotalPulls = 0
	cumulativeReward = 0
	for explore in range(k):
		It = explore
		reward = select_arm(It)
		values[It] += reward
		loss[It] += 1-reward
		counts[It] +=1
		cumulativeReward += reward

	mu_hat = values/counts
	regret = np.zeros(total_rounds)

	for t in range(k, total_rounds):
		Aj = np.argmax(mu_hat + np.sqrt(alpha*np.log(t)/counts))
		# Aj
		reward = select_arm(Aj)
		values[Aj] += reward
		loss[Aj] += 1-reward
		counts[Aj] += 1
		mu_hat = values/counts

		cumulativeReward += reward
		index = np.argmax(counts)

		regret[t] += regret[t-1] + max_mu -(mu_hat[Aj])

	return regret

def thompson(num_arms, total_rounds):
	k = num_arms
	eps = [0]*k
	counts = np.zeros(k)
	values = np.zeros(k)
	loss = np.zeros(k)
	s = np.zeros(k)
	f = np.zeros(k)
	theta = np.zeros(k)
	regret = np.zeros(total_rounds)
	max_mu = p
	for t in range(total_rounds):
		for i in range(k):
			a = s[i] + 1
			b = f[i] + 1
			theta[i] = np.random.beta(a, b)
		# theta
		It = np.argmax(theta)
		# It
		reward = select_arm(It)
		if(reward == 1):
			s[It] += 1
		else:
			f[It] += 1
		mu_hat = s/(s+f)
		## mu_hat
		regret[t] += regret[t-1] + max_mu - (mu_hat[It])
		
	# mu_hat
	# min_mu
	return regret


cum_regret1 = []
cum_regret2 = []
cum_regret3 = []
cum_regret4 = []

for i in range(num_sample_paths):
	cum_regret1.append(e_greedy(T))
	cum_regret2.append(ucb(k,T))
	cum_regret3.append(kl_ucb(k,T))
	cum_regret4.append(thompson(k,T))

# cum_regret
#time = []
#for t in range(T):
#	time.append(t)
#############################################################################
time = 100*np.arange(T/100)
freedom_degree = len(time) - 1 #250 - 1
regret_mean = []
regret_err = []


R = np.zeros((num_sample_paths,T/100))
for j in range(num_sample_paths):
	for i in range(T/100):
		R[j][i] = cum_regret1[j][100*(i+1)-1]

regret_mean = R.mean(axis=0)
for i in range(T/100):
	regret_err.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(R[:,i]))
	regret_err[i] = regret_err[i]*(i%25==0)

colors = list("rgbcmyk")
shape = ['--^', '--d', '--v', '--x']

plt.errorbar(time, regret_mean, regret_err, color=colors[0])

plt.plot(time, regret_mean, colors[0] + shape[0], label='e-greedy')
################################################################################
regret_mean = []
regret_err = []

R = np.zeros((num_sample_paths,T/100))
for j in range(num_sample_paths):
	for i in range(T/100):
		R[j][i] = cum_regret2[j][100*(i+1)-1]

regret_mean = R.mean(axis=0)
for i in range(T/100):
	regret_err.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(R[:,i]))
	regret_err[i] = regret_err[i]*(i%25==0)
colors = list("rgbcmyk")
shape = ['--^', '--d', '--v', '--x']

plt.errorbar(time, regret_mean, regret_err, color=colors[1])

plt.plot(time, regret_mean, colors[1] + shape[1], label='ucb')
#############################################################################

regret_mean = []
regret_err = []

R = np.zeros((num_sample_paths,T/100))
for j in range(num_sample_paths):
	for i in range(T/100):
		R[j][i] = cum_regret3[j][100*(i+1)-1]

regret_mean = R.mean(axis=0)
for i in range(T/100):
	regret_err.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(R[:,i]))
	regret_err[i] = regret_err[i]*(i%25==0)

colors = list("rgbcmyk")
shape = ['--^', '--d', '--v', '--x']

plt.errorbar(time, regret_mean, regret_err, color=colors[2])
plt.plot(time, regret_mean, colors[2] + shape[2], label='kl_ucb')
#############################################################################

regret_mean = []
regret_err = []

R = np.zeros((num_sample_paths,T/100))
for j in range(num_sample_paths):
	for i in range(T/100):
		R[j][i] = cum_regret4[j][100*(i+1)-1]

regret_mean = R.mean(axis=0)
for i in range(T/100):
	regret_err.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(R[:,i]))
	regret_err[i] = regret_err[i]*(i%25==0)

colors = list("rgbcmyk")
shape = ['--^', '--d', '--v', '--x']

plt.errorbar(time, regret_mean, regret_err, color=colors[3])

plt.plot(time, regret_mean, colors[3] + shape[3], label='thompson')
#############################################################################


plt.legend(loc='upper right', numpoints=1)
plt.title("Pseudo Regret vs Num rounds for k =20 and 20 Sample paths")
plt.xlabel("No of rounds")
plt.ylabel("Pseudo Regret")
plt.show()

