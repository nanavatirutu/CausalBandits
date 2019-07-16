from bandits import Bandits
from thompson_sampling import ThompsonSampling


if __name__ == "__main__":
	print("initializing bandits...")
	# make the bandits - these are the true probabilities
	bandits = Bandits([0.8, 0.6, 0.4])

	print("initializing thompson sampling agents to learn the parameters...")
	# train the model
	thompson_sampling = ThompsonSampling(bandits)
	print("training the agents...")
	thompson_sampling.train()
	print("training successful...")
