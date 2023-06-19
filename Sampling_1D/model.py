import numpy as np
import numpy.random as rd

def V(x):
	return 0.5*x**2

def mu(x):
	return np.exp(-V(x))

# State space
s_left = -6
s_right = 6
ds = 12.0/1000.0
state_space = np.linspace(-6.0, 6.0, 1001)
state_coor_to_index = lambda x: round( (x + 6.0)/ds ) 
state_index_to_coor = lambda s: -6.0 + s*ds

# Action space
Dt = 0.01
da = 6.0/100
action_space = np.sqrt(2.0*Dt)*np.linspace(-3.0, 3.0, 101)
action_coor_to_index = lambda y: round( (y/np.sqrt(2.0*Dt) + 3.0)/da )
action_index_to_coor = lambda a: np.sqrt(2.0*Dt)*(-3.0 + a*da)

# Rewards
rewards = mu(state_space)

def step(state, action):
	#print('step', state, action)
	# Map index states and actions to coordinates
	coor_state = state_index_to_coor(state)
	coor_action = action_index_to_coor(action)

	# Calculate the next state
	next_state = coor_state + coor_action

	# Map the coordinate state back to indices
	next_state = state_coor_to_index(next_state)
	if next_state > 1000:
		next_state = 0 + (next_state - 1000)
	elif next_state < 0:
		next_state = 1000 + next_state

	# Return
	reward = rewards[next_state]
	return next_state, reward, (next_state >= 498 and next_state <= 502)

def testStateMapping(): # Opnieuw testen
	print('State Test')
	coor_state = -6.0 + 154.0*ds
	index_state = state_coor_to_index(coor_state)
	new_coor_state = state_index_to_coor(index_state)

	print(coor_state, index_state)
	print(coor_state, new_coor_state)

def testActionMapping(): # Opnieuw testen
	print('\nAction Test')
	coor_action = np.sqrt(2.0*Dt)*(-3.0 + 51.0*da)
	index_action = action_coor_to_index(coor_action)
	new_coor_action = action_index_to_coor(index_action)

	print(coor_action, index_action)
	print(coor_action, new_coor_action)

if __name__ == '__main__':
	testStateMapping()
	testActionMapping()
