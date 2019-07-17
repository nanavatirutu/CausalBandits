from bandits import Bandits
from thompson_sampling import ThompsonSampling
from visualize import visualize_plot

def regret(probs, choices):
	w_opt = max(probs)
	return (w_opt - probs[choices.astype(int)]).cumsum()


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
	print("Visualizing regret...")
	visualize_plot(thompson_sampling.regret(), "# of trials", "Cumulative regret")

	print("Visualizing probability of choosing best arm.. ")
	visualize_plot(thompson_sampling.prob_best_arm(), "# of trials", "Probability of choosing best arm")








