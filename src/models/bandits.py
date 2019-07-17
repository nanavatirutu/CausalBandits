from pyro import distributions as dist
from pyro import sample
from torch import tensor
class Bandits(object):
	def __init__(self, true_probabilities):
		self.true_probabilities = true_probabilities
		self.best = self.true_probabilities.index(max(self.true_probabilities))  # get the best arm

	# action of pulling each slot machine based on true probability of each bandit
	def pull(self, i):
		if i >= 0 and i < len(self.true_probabilities):
			y = sample("result", dist.Bernoulli(tensor(self.true_probabilities[i])))
			return y.item()
		else:
			print("Error: invalid choice")
			return -1

	# get length of bandits
	def __len__(self):
		return len(self.true_probabilities)