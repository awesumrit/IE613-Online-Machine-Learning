import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.stats as ss

T = 100000

p = 0.5
delta = 0.1
K = 2 #arms
d = K
n_arms = K
val = np.arange(0.1, 2.11, 0.2)
neta = []
for c in val:
	neta.append(np.sqrt(2*np.log(K)/(K*T))*c)

total_prediction_rounds = T

class Nature:
	def __init__(self, num_experts):
		self.num_experts = num_experts
		self.experts = []
		for expert_id in range(K):
			self.experts.append(Experts(expert_id))

	def get_observation(self, roundNumber):
		observation = []
		for expert in self.experts:
			observation.append(expert.get_advice(roundNumber))
		return observation

	def get_expert_losses(self, label, round_number):
		expert_losses = []
		for expert in self.experts:
			expert_losses.append(expert.expert_loss(label, round_number))
		return expert_losses


class AdverserialNature(Nature):
	def get_label(self, weights, observation):
		#Uses current weights and expert advice to give label
		#as of what your expected prediction might be
		#Predict according to weighted majority and return prediction
		prediction = np.sign(np.sum(np.multiply(weights, observation)))
		return prediction


#Class characterizing the different kinds of experts
class Experts:
	def __init__(self, expert_id):
		self.expert_id = expert_id
	def get_advice(self, round_number):
		if self.expert_id < K-1 :
			return np.random.binomial(1, p)
		elif self.expert_id == K-1 :
			return np.random.binomial(1, p+0.05)

	def expert_loss(self, true_label, round_number):
		return 0 if(true_label - self.get_advice(round_number)) == 0 else 1


class Learner:
	def __init__(self,nature, prediction_type, ita, adverserial=False): 
		self.adverserial = adverserial
		self.nature = nature
		self.prediction_type = prediction_type
		#Learning parameters
		self.ita = ita

	#Generic function that can incorporate different types of learners
	def learn(self, total_prediction_rounds):
		#The general online learning paradigm is followed
		#Nature sends observation, learner makes prediction according 
		#to prediction scheme
		regret = 0
		regretp = 0
		regretix = 0
		if self.prediction_type == "Exp3":
			regret, regretp, regretix = self.Exp3(self.ita, total_prediction_rounds)


		return regret, regretp, regretix

	def Exp3(self, ita, t):
		n_arms = K
		Loss = np.zeros(K)
		regret = 0
		L = np.zeros(K)
		mu = 0
		mup = 0
		muix = 0

		#min_mu = T/4*(p)+(3*T/4)*(p+0.05)
		min_mu = T*p
		x = np.zeros(K)
		xp = np.zeros(K)
		xix = np.zeros(K)
		probs = np.zeros(K)
		probsp = np.zeros(K)
		probsix = np.zeros(K)
		s = np.zeros(K)
		sp = np.zeros(K)
		six = np.zeros(K)

		gammap = K*ita	
		gammaix = K*ita/2.0
		beta = ita


		for round_number in range(total_prediction_rounds):

			probs = np.exp(ita*s) / np.sum(np.exp(ita*s))
			probsp = (1.0-gammap)*np.exp(ita*sp)/np.sum(np.exp(ita*sp)) + gammap/K
			probsix = np.exp(ita*six) / np.sum(np.exp(ita*six))

			It = np.random.choice(np.arange(0,K), p=probs)
			Itp = np.random.choice(np.arange(0,K), p=probsp)
			Itix = np.random.choice(np.arange(0,K), p=probsix)

			for idx in range(K):
				Loss[idx] = self.nature.experts[idx].get_advice(round_number)
			Xt = Loss[It]
			Xtp = Loss[Itp]
			Xtix = Loss[Itix]

			x = np.array([1-Xt/probs[i] if i == It else 1 for i in range(K) ])
			xp = np.array([(1-Xtp+beta)/probsp[i] if i == Itp else beta/probsp[i] for i in range(K) ])
			xix = np.array([(1-Xtix)/(gammaix+probsix[i]) if i == Itix else 0 for i in  range(K)])

			s = s + x 
			sp = sp + xp 
			six = six + xix 

			if(It<K-1):
				update = p
			elif(It==K-1):
				update = p+0.05

			if(Itp<K-1):
				updatep = p
			elif(Itp==K-1):
				updatep = p+0.05

			if(Itix<K-1):
				updateix = p
			elif(Itix==K-1):
				updateix = p+0.05

			mu = mu + update
			mup = mup + updatep
			muix = muix + updateix
		regret = mu - min_mu
		regretp = mup - min_mu
		regretix = muix - min_mu
		return regret, regretp, regretix

def main():

	total_prediction_rounds = T
	#Choose the kind of nature and nuber of experts in that nature (d no of experts)
	exp3_regret = [[] for i in range(len(neta))]# = [[...], ..., [etaValue_i, regretSamplePath_i1, ..., regretSamplePath_iC], ..., [...]]
	exp3p_regret = [[] for i in range(len(neta))]
	exp3ix_regret = [[] for i in range(len(neta))]
	nature = AdverserialNature(d)
	paths = 1
	for i in range(len(neta)):
		ita = neta[i]
		avg_regret = np.zeros(paths)
		avg_regretp = np.zeros(paths)
		avg_regretix = np.zeros(paths)

		exp3_regret[i].append(ita)
		exp3p_regret[i].append(ita)
		exp3ix_regret[i].append(ita)
		for samplePath in range(paths):
			#Choose kind of learner and whether nature is adverserial
			Exp3_learner = Learner(nature, "Exp3", ita, True)
			avg_regret[samplePath], avg_regretp[samplePath], avg_regretix[samplePath] = Exp3_learner.learn(total_prediction_rounds)

			exp3_regret[i].append(avg_regret[samplePath])
			exp3p_regret[i].append(avg_regretp[samplePath])
			exp3ix_regret[i].append(avg_regretix[samplePath])
			
		print "Regret = {}".format(avg_regret) 
		print "Regret = {}".format(avg_regretp) 
		print "Regret = {}".format(avg_regretix)  

	eta = []
	regret_mean = []
	regret_err = []
	freedom_degree = len(exp3_regret[0]) - 2
	for regret in exp3_regret:
		eta.append(regret[0])
		regret_mean.append(np.mean(regret[1:]))
		regret_err.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[1:]))

	colors = list("rgbcmyk")
	shape = ['--^', '--d', '--v']
	plt.errorbar(eta, regret_mean, regret_err, color=colors[0])
	plt.plot(eta, regret_mean, colors[0] + shape[0], label='EXP3')


	eta = []
	regret_mean = []
	regret_err = []
	freedom_degree = len(exp3p_regret[0]) - 2
	for regret in exp3p_regret:
		eta.append(regret[0])
		regret_mean.append(np.mean(regret[1:]))
		regret_err.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[1:]))

	colors = list("rgbcmyk")
	shape = ['--^', '--d', '--v']
	plt.errorbar(eta, regret_mean, regret_err, color=colors[1])
	plt.plot(eta, regret_mean, colors[1] + shape[1], label='EXP3p')

	eta = []
	regret_mean = []
	regret_err = []
	freedom_degree = len(exp3ix_regret[0]) - 2
	for regret in exp3ix_regret:
		eta.append(regret[0])
		regret_mean.append(np.mean(regret[1:]))
		regret_err.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(regret[1:]))

	colors = list("rgbcmyk")
	shape = ['--^', '--d', '--v']
	plt.errorbar(eta, regret_mean, regret_err, color=colors[2])
	plt.plot(eta, regret_mean, colors[2] + shape[2], label='EXP3ix')


	plt.legend(loc='upper right', numpoints=1)
	plt.title("Pseudo Regret vs Learning Rate for T = 10^5 and 50 Sample paths")
	plt.xlabel("Learning Rate")
	plt.ylabel("Pseudo Regret")
	plt.savefig("Q2.png", bbox_inches='tight')
	plt.close()


if __name__ == "__main__":
	main()
