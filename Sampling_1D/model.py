import numpy as np
import numpy.random as rd

def V(x):
	return 0.5*x**2

def mu(x):
	return np.exp(-V(x))

# State space
s_left = -6
s_right = 6
ds = 6.0/500.0
state_space = np.linspace(-6.0, 6.0, 1001)
state_coor_to_index = lambda x: round( (x+6)*1000.0 )
state_index_to_coor = lambda i: -6.0 + i/1000.0

# Action space
Dt = 0.01
action_space = np.sqrt(2.0*Dt)*np.linspace(-3.0, 3.0, 101)
action_coor_to_index = lambda a: round((a/np.sqrt(2.0*Dt) + 3.0)*100.0)
action_index_to_coor = lambda i: np.sqrt(2.0*Dt)*(-3.0 + i/100.0)

# Rewards
rewards = mu(state_space)

def step(state, action):
	# Map index states and actions to coordinates
	coor_state = state_index_to_coor(state)
	coor_action = action_index_to_coor(action)

	# Calculate the next state
	next_state = coor_state + coo_action

	# Map the coordinate state back to indices
	next_state = state_coor_to_index(next_state)

	# Return
	reward = rewards[next_state]
	return next_state, reward, (next_state == 500)

def testStateMapping(): # Works!
	print('State Test')
	coor_state = -6.0 + 154.0/1000.0
	index_state = state_coor_to_index(coor_state)
	new_coor_state = state_index_to_coor(index_state)

	print(coor_state, index_state)
	print(coor_state, new_coor_state)

def testActionMapping(): # Works!
	print('\nAction Test')
	coor_action = np.sqrt(2.0*Dt)*(-3.0 + 51.0/100.0)
	index_action = action_coor_to_index(coor_action)
	new_coor_action = action_index_to_coor(index_action)

	print(coor_action, index_action)
	print(coor_action, new_coor_action)

if __name__ == '__main__':
	testStateMapping()
	testActionMapping()
