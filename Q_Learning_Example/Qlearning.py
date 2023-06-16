import numpy as np
import matplotlib.pyplot as plt
from bidict import bidict
from scipy.stats import randint
import statistics

# Rewards
rewards = np.array([0, 1, 2, 10, 0, 3, -10, 1, 2, 2, 3])

# Create state/cartesian mapping
m = {0: (0,2), 1: (1,2), 2: (2,2), 3: (3,2), 4: (0,1), 5: (2,1), 6: (3, 1), 7: (0,0), 8: (1,0), 9: (2,0), 10: (3, 0)}
s_to_cart = bidict(m)
cart_to_s = s_to_cart.inverse

def sample_action(state, q_matrix):
	while True:
		u = randint.rvs(0, 4)
		action = u
		if q_matrix[state, action] == -np.inf:
			continue
		else:
			return action

def step(state, action, rewards, s_cart, cart_s):
	cart_state = s_cart[state]
	if action == 0:
		cart_state = (cart_state[0], cart_state[1]+1)
	elif action == 1:
		cart_state = (cart_state[0], cart_state[1]-1)
	elif action == 2:
		cart_state = (cart_state[0]-1, cart_state[1])
	else:
		cart_state = (cart_state[0]+1, cart_state[1])

	new_state = cart_s[cart_state]
	r = rewards[new_state]
	done = (new_state == 3)

	return new_state, r, done

def bellman_stepper():
	# Parameters
	n_episodes = 1500
	max_iter_episode = 100
	gamma = 0.95
	alpha = 0.1

	exploration_probability = 1.0 # Changes per episode
	min_exploration_probability = 0.01
	lam = 0.1

	# Create q-matrix and determine boundaries
	q_matrix = np.zeros((11, 4))
	q_matrix[0,0] = q_matrix[0,2] = -np.inf
	q_matrix[1,0] = q_matrix[1,1] = -np.inf
	q_matrix[2,0] = -np.inf
	q_matrix[3,0] = q_matrix[3,3] = -np.inf
	q_matrix[4,2] = q_matrix[4,3] = -np.inf
	q_matrix[5,2] = -np.inf
	q_matrix[6,3] = -np.inf
	q_matrix[7,1] = q_matrix[7,2] = -np.inf
	q_matrix[8,0] = q_matrix[8,1] = -np.inf
	q_matrix[9,1] = -np.inf
	q_matrix[10,1] = q_matrix[10,3] = -np.inf

	total_rewards_episode = list()
	for e in range(n_episodes):
		current_state = 7
		done = False

		#sum the rewards that the agent gets from the environment
		total_episode_reward = rewards[current_state]
		episode_states = [current_state]

		for i in range(max_iter_episode): 
			# we sample a float from a uniform distribution over 0 and 1
			# if the sampled float is less than the exploration probability
			#     the agent selects a random action
			# else
			#     he exploits his knowledge using the Bellman equation

			if np.random.uniform(0.0, 1.0) <= exploration_probability:
				action = sample_action(current_state, q_matrix)
			else:
				action = np.argmax(q_matrix[current_state,:])

			# The environment runs the chosen action and returns
			# the next state, a reward and true if the epiosed is ended.
			next_state, reward, done = step(current_state, action, rewards, s_to_cart, cart_to_s)

			# We update our Q-table using the Q-learning iteration
			q_matrix[current_state, action] = (1.0-alpha) * q_matrix[current_state, action] + alpha*(reward + gamma*max(q_matrix[next_state,:]))
			total_episode_reward = total_episode_reward + reward

			# If the episode is finished, we leave the for loop
			current_state = next_state
			episode_states.append(current_state)
			if done:
				break
		averaged_episode_reward = total_episode_reward / len(episode_states)
		#print('Episode #', e, 'States:', episode_states, averaged_episode_reward, total_episode_reward)

		#We update the exploration proba using exponential decay formula 
		exploration_probability = max(min_exploration_probability, np.exp(-lam*e))
		total_rewards_episode.append(averaged_episode_reward)

		if e % 100 == 0:
			print('Episode #',e,':', statistics.mean(total_rewards_episode))

		np.save('Q_Table_2.npy', q_matrix)

def followPath():
	Q_Table = np.load('Q_Table_2.npy')
	if Q_Table.shape[0] < Q_Table.shape[1]:
		Q_Table = np.transpose(Q_Table)

	# Continuation
	state = 7
	states = [state]
	rs = rewards[state]
	while True:
		print(state)
		action = np.argmax(Q_Table[state,:])
		state, reward, done = step(state, action, rewards, s_to_cart, cart_to_s)

		states.append(state)
		rs += rewards[state]
		if done:
			print('Path', states)
			print('Total Reward', rs / len(states))
			print('Done')
			break

def debug_QTable():
	q1 = np.load('Q_Table.npy')
	q2 = np.load('Q_Table_2.npy')

	print('q1\n', q1,'\n')
	print('q2\n', q2)


if __name__ == '__main__':
	bellman_stepper()


