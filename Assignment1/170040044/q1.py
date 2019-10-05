import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import scipy.stats as ss

T = 100000
d = 10
p = 0.5
delta = 0.1

val = np.arange(0.1, 2.11, 0.2)
neta = []
for c in val:
	neta.append(math.sqrt(2*math.log(d)/T)*c)

class Nature:
	def __init__(self, num_experts):
		self.num_experts = num_experts
		self.experts = []
		for expert_id in range(10):
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
		if self.expert_id < 8 :
			return np.random.binomial(1, p)
		elif self.expert_id == 8 :
			return np.random.binomial(1, p-delta)
		elif self.expert_id == 9 :
			return np.random.binomial(1, p+delta) if (round_number<=T/2) else np.random.binomial(1, p-2*delta)
	def expert_loss(self, true_label, round_number):
		return 0 if(true_label - self.get_advice(round_number)) == 0 else 1


class Learner:
	def __init__(self,nature, prediction_type, eta, adverserial=False): 
		self.adverserial = adverserial
		self.nature = nature
		self.weights = [1]*self.nature.num_experts
		self.prediction_type = prediction_type
		#Learning parameters
		self.eta = eta

	#Generic function that can incorporate different types of learners
	def learn(self, total_prediction_rounds):
		#The general online learning paradigm is followed
		#Nature sends observation, learner makes prediction according 
		#to prediction scheme (WMA)
		total_learner_loss = 0
		regret_t = 0
		learner_loss = 0
		if self.prediction_type == "weighted_majority":
			regret_t, learner_loss = self.weighted_majority(self.eta, total_prediction_rounds)

		return regret_t, learner_loss

	def weighted_majority(self, etta, T):
		W_hat = np.ones(10)
	
		Loss = np.zeros(10)
		regret = 0
		L = np.zeros(10)
		mu = 0

		for round_number in range(T):
			W = W_hat/W_hat.sum()

			for idx in range(10):
				Loss[idx] = self.nature.experts[idx].get_advice(round_number)
			L = L + Loss
			W_hat = W_hat * np.exp(-etta*Loss)
			mu = mu + np.dot(Loss, W)

		regret = mu - np.min(L)
	
		return regret, L



def main():

	total_prediction_rounds = T
	#Choose the kind of nature and nuber of experts in that nature (d no of experts)
	exp3_regret = [[] for i in range(len(neta))]# = [[...], ..., [etaValue_i, regretSamplePath_i1, ..., regretSamplePath_iC], ..., [...]]
	nature = AdverserialNature(d)
	Episode_learner_loss = []


	for i in range(len(neta)):
		ita = neta[i]
		avg_regret = np.zeros(20)
		exp3_regret[i].append(ita)
		for samplePath in range(20):
			#Choose kind of learner and whether nature is adverserial
			Exp3_learner = Learner(nature, "weighted_majority", ita, True)
			avg_regret[samplePath], avg_learner_loss = Exp3_learner.learn(total_prediction_rounds)
			exp3_regret[i].append(avg_regret[samplePath])
			
		print "Regret = {}".format(avg_regret)  
		

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
	plt.plot(eta, regret_mean, colors[0] + shape[0], label='weighted_majority')
	#plt.show()

	plt.legend(loc='upper right', numpoints=1)
	plt.title("Pseudo Regret vs Learning Rate for T = 10^5 and 20 Sample paths")
	plt.xlabel("Learning Rate")
	plt.ylabel("Pseudo Regret")
	plt.savefig("Q1.png", bbox_inches='tight')
	plt.close()

if __name__ == "__main__":
	main()
