from model import *
import matplotlib.pyplot as plt
import statistics

def Q_learning():
	Q_Table = np.zeros((state_space.size, action_space.size))

	n_episodes = 100000
	max_steps = 10**5
	gamma = 0.95
	lr = 0.1

	# Exploration probability decresses per iteration
	exploration_probability = 1.0 # Changes per episode
	min_exploration_probability = 0.01
	lam = 0.1

	rewards_per_episode = list()
	for episode in range(1, n_episodes+1):
		state = rd.randint(0,1001) # state = state_coor_to_index(-2.0)

		done = False
		episode_reward = 0.0
		episode_states = [state]
		for i in range(max_steps):
			# The environment runs the chosen action and returns
			# the next state, a reward and true if the epiosed is ended.
			if rd.uniform(0.0, 1.0) <= exploration_probability:
				action = rd.randint(0,101)
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
		#print('Episode #', episode, 'States:', episode_states, averaged_episode_reward, episode_reward)
		#print('Episode #', episode, averaged_episode_reward)

		#We update the exploration proba using exponential decay formula 
		exploration_probability = max(min_exploration_probability, np.exp(-lam*episode))
		rewards_per_episode.append(averaged_episode_reward)

		if episode % 100 == 0:
			print('Episode #',episode,':', statistics.mean(rewards_per_episode))

	print('Storing Q_Table.')
	np.save('Q_Table_Random.npy', Q_Table)

def _followPath(Q_Table, state):
	all_states = [state]
	total_discounted_reward = rewards[state]
	discounter = 1.0
	gamma = 0.95
	while True:
		action = np.argmax(Q_Table[state,:])
		new_state, reward, _ = step(state, action)

		if new_state == state:
			break

		state = new_state
		all_states.append(state)
		total_discounted_reward += gamma**discounter * reward
		discounter += 1

	return all_states, total_discounted_reward

def plotValueFunction():
	Q_Table = np.load('Q_Table_Random.npy')

	state_space = np.linspace(-6.0, 6.0, 1001)
	values = []
	for i in range(len(state_space)):
		state = state_coor_to_index(state_space[i])

		all_states, total_discounted_reward = _followPath(Q_Table, state)
		values.append(total_discounted_reward)

	values = np.array(values)
	values = 0.5*(values + values[::-1])

	# Filter frequency 1000
	fft_values = np.fft.fft(values)
	fft_values[50:1000] = 0.0
	ifft_values = np.fft.ifft(fft_values)

	plt.plot(state_space, values)
	plt.xlabel('State')
	plt.ylabel('Value Function')

	plt.figure()
	plt.plot(state_space, ifft_values)
	plt.plot(np.linspace(-25.0, 25.0, 1001), 6.0*np.exp(-0.5*np.linspace(-25.0, 25.0, 1001)**2/20.0))
	plt.xlabel('State')
	plt.ylabel('Value Function')

	plt.show()

def followOptimalPath():
	Q_Table  = np.load('Q_Table_Random.npy')

	# Generate random initial conditioin and see if it converges
	# under the policy of Q.
	state = rd.randint(0,1001)
	all_states, total_discounted_reward = _followPath(Q_Table, state)

	print('States', all_states)
	print('Cumulated Reward', total_discounted_reward)


if __name__ == '__main__':
	plotValueFunction()