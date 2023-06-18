from model import *
import matplotlib.pyplot as plt

def Q_learning():
	Q_Table = np.zeros((state_space.size, action_space.size))

	n_episodes = 1
	max_steps = 10**5
	gamma = 0.95
	lr = 0.1

	# Exploration probability decresses per iteration
	exploration_probability = 1.0 # Changes per episode
	min_exploration_probability = 0.01
	lam = 0.1

	rewards_per_episode = list()
	for episode in range(1, n_episodes+1):
		state = -2.0

		done = False
		episode_reward = 0.0
		episode_states = [state]
		for i in range(max_steps):
			# The environment runs the chosen action and returns
			# the next state, a reward and true if the epiosed is ended.
			if rd.uniform(0.0, 1.0) <= exploration_probability:
				action = action_space[rd.randint(0,101)]
			else:
				action = np.argmax(Q_Table[state,:])
			next_state, reward, done = step(state, action)

			# We update our Q-table using the Q-learning iteration
			Q_Table[state, action] = (1.0-lr) * Q_Table[state, action] + lr*(reward + gamma*max(Q_Table[next_state,:]))
			episode_reward = episode_reward + reward

			# If the episode is finished, we leave the for loop
			state = next_state
			episode_states.append(state)
			if done:
				break


		averaged_episode_reward = episode_reward / len(episode_states)
		print('Episode #', e, 'States:', episode_states, averaged_episode_reward, total_episode_reward)

		#We update the exploration proba using exponential decay formula 
		exploration_probability = max(min_exploration_probability, np.exp(-lam*episode))
		total_rewards_episode.append(averaged_episode_reward)

		if episode % 100 == 0:
			print('Episode #',episode,':', statistics.mean(rewards_per_episode))


if __name__ == '__main__':
	Q_learning()