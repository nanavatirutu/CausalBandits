from numpy import zeros
from numpy import ones
from pyro import sample
from pyro import distributions as dist
class ThompsonSampling(object):
	def __init__(self, bandits):
		# get bandit info
		self.bandits = bandits
		self.n_bandits = len(bandits)

		# learning probablities
		# the learners would be a beta distribution corresponding to each bandit
		self.learnt_params = ones((self.n_bandits,2))

		# strategy evaluation variables
		self.trials = zeros(self.n_bandits)
		self.wins = zeros(self.n_bandits)
		self.N = 0
		self.choices = []
		self.score = []

	def model(self):
		values = [sample("values", dist.Beta(*self.learnt_params[i])).item() for i in range(self.n_bandits)]
		choice = values.index(max(values))
		result = self.bandits.pull(choice)
		return choice, result

	def train(self, n_trials=2000):
		# run the simulation
		for i in range(n_trials):
			choice, result = self.model()
			if result == 1:
				self.learnt_params[choice][0] += 1
			else:
				self.learnt_params[choice][1] += 1
			self.wins[choice] += result
			self.trials[choice] += 1
			self.score.append(result)
			self.N += 1
			self.choices.append(choice)
		return

