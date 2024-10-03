import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

'''
# CS715: Advanced Machine Learning
## Assignment: Q-Learning for Warehouse Robot Pathfinding
### Introduction
In this assignment, we have implemented a Q-learning-based solution to optimize the path of a warehouse robot navigating through an environment with various obstacles and goals. The environment is modeled after a warehouse layout, where the robot's task is to find the shortest path from any given starting point to a designated item packaging area.

### Problem Description
The problem simulates a warehouse environment with multiple states (locations) and obstacles (storage areas). The robot must learn the optimal path to the packaging area using Q-learning. The environment is structured as an 11x11 grid representing different locations:

- Aisles (white squares): Locations the robot can navigate.
- Storage areas (black squares): Obstacles that the robot must avoid.
- Packaging area (green square): The target destination for the robot.
- The goal is to use reinforcement learning (specifically, Q-learning) to help the robot learn the most efficient path to the packaging area while avoiding obstacles.

### Q-Learning Implementation
The environment consists of states (locations within the grid), actions (movement directions), and rewards:

- States: All possible grid positions within the warehouse, represented as an 11x11 matrix.
- Actions: The robot can move in four possible directions: up, right, down, and left.
- Rewards: The packaging area provides a positive reward (+100), while other navigable locations have a small negative reward (-1) to encourage the robot to find the shortest path. All storage areas have a high negative reward (-100) to discourage the robot from entering them.
'''

environment_rows = 11
environment_columns = 11

q_values = np.zeros((environment_rows, environment_columns, 4))
actions = ['up', 'right', 'down', 'left']

rewards = np.full((environment_rows, environment_columns), -100.)
rewards[0, 5] = 100. #set the reward for the packaging area (i.e., the goal) to 100

#define aisle locations (i.e., white squares) for rows 1 through 9
aisles = {} #store locations in a dictionary
aisles[1] = [i for i in range(1, 10)]
aisles[2] = [1, 9]
aisles[3] = [i for i in range(1, 8)] + [9]
# aisles[3].append(9)
aisles[4] = [3, 7, 9]
aisles[5] = [i for i in range(1,10)]
aisles[6] = [2, 5, 8]
aisles[7] = [i for i in range(1, 10)]
aisles[8] = [1, 9]
aisles[9] = [i for i in range(1,10)]

#set the rewards for all aisle locations (i.e., white squares)
for row_index in range(1, 10):
  for column_index in aisles[row_index]:
    rewards[row_index, column_index] = -1.

#print rewards matrix
for row in rewards:
  print(row)

#define a function that determines if the specified location is a terminal state
def is_terminal_state(current_row_index, current_column_index):
  if rewards[current_row_index, current_column_index] == -1.:
    return False
  else:
    return True

#define a function that will choose a random, non-terminal starting location
def get_starting_location():
  #get a random row and column index
  current_row_index = np.random.randint(environment_rows)
  current_column_index = np.random.randint(environment_columns)
  #continue choosing random row and column indexes until a non-terminal state is identified
  #(i.e., until the chosen state is a 'white square').
  while is_terminal_state(current_row_index, current_column_index):
    current_row_index = np.random.randint(environment_rows)
    current_column_index = np.random.randint(environment_columns)
  return current_row_index, current_column_index

#define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
def get_next_action(current_row_index, current_column_index, epsilon):
  #if a randomly chosen value between 0 and 1 is less than epsilon,
  #then choose the most promising value from the Q-table for this state.
  if np.random.random() < epsilon:
    return np.argmax(q_values[current_row_index, current_column_index])
  else: #choose a random action
    return np.random.randint(4)

#define a function that will get the next location based on the chosen action
def get_next_location(current_row_index, current_column_index, action_index):
  new_row_index = current_row_index
  new_column_index = current_column_index
  if actions[action_index] == 'up' and current_row_index > 0:
    new_row_index -= 1
  elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
    new_column_index += 1
  elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
    new_row_index += 1
  elif actions[action_index] == 'left' and current_column_index > 0:
    new_column_index -= 1
  return new_row_index, new_column_index

#Define a function that will get the shortest path between any location within the warehouse that
#the robot is allowed to travel and the item packaging location.
def get_shortest_path(start_row_index, start_column_index):
  #return immediately if this is an invalid starting location
  if is_terminal_state(start_row_index, start_column_index):
    return []
  else: #if this is a 'legal' starting location
    current_row_index, current_column_index = start_row_index, start_column_index
    shortest_path = []
    shortest_path.append([current_row_index, current_column_index])
    #continue moving along the path until we reach the goal (i.e., the item packaging location)
    while not is_terminal_state(current_row_index, current_column_index):
      #get the best action to take
      action_index = get_next_action(current_row_index, current_column_index, 1.)
      #move to the next location on the path, and add the new location to the list
      current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
      shortest_path.append([current_row_index, current_column_index])
    return shortest_path

#define training parameters
st.sidebar.write('## Define Learning Parameters:')
epsilon = st.sidebar.slider('epsilon', min_value=0.3, max_value=1.0, step=0.1, value=0.9) # 0.9 #the percentage of time when we should take the best action (instead of a random action)
discount_factor = st.sidebar.slider('discount_factor', min_value=0.3, max_value=1.0, step=0.1, value=0.9) # 0.9 #discount factor for future rewards
learning_rate = st.sidebar.slider('learning_rate', min_value=0.3, max_value=1.0, step=0.1, value=0.9) # 0.9 #the rate at which the AI agent should learn

#run through 1000 training episodes
for episode in range(1000):
  #get the starting location for this episode
  row_index, column_index = get_starting_location()

  #continue taking actions (i.e., moving) until we reach a terminal state
  #(i.e., until we reach the item packaging area or crash into an item storage location)
  while not is_terminal_state(row_index, column_index):
    #choose which action to take (i.e., where to move next)
    action_index = get_next_action(row_index, column_index, epsilon)

    #perform the chosen action, and transition to the next state (i.e., move to the next location)
    old_row_index, old_column_index = row_index, column_index #store the old row and column indexes
    row_index, column_index = get_next_location(row_index, column_index, action_index)

    #receive the reward for moving to the new state, and calculate the temporal difference
    reward = rewards[row_index, column_index]
    old_q_value = q_values[old_row_index, old_column_index, action_index]
    temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value

    #update the Q-value for the previous state and action pair
    new_q_value = old_q_value + (learning_rate * temporal_difference)
    q_values[old_row_index, old_column_index, action_index] = new_q_value

print('Training complete!')

def plot_environment(path=None):
    fig, ax = plt.subplots(figsize=(2,2))
    # Plot warehouse layout
    for row in range(environment_rows):
        for col in range(environment_columns):
            if rewards[row, col] == -100.:
                color = 'black'  # Storage locations
            elif rewards[row, col] == 100.:
                color = 'green'  # Packaging area
            else:
                color = 'white'  # Aisles
            ax.add_patch(plt.Rectangle((col, environment_rows - row - 1), 1, 1, color=color, edgecolor='gray'))
    
    # Plot path if available
    if path:
        for (row, col) in path:
            ax.add_patch(plt.Rectangle((col, environment_rows - row - 1), 1, 1, color='yellow', edgecolor='gray', alpha=0.5))
    
    plt.xlim(0, environment_columns)
    plt.ylim(0, environment_rows)
    ax.set_aspect('equal')
    ax.axis('off')
    st.pyplot(fig, use_container_width=False)

"""### Overview of the Environment
"""

plot_environment()
st.write('## Select Source Cordinates:')
y = st.slider('Select X of starting point', min_value=0, max_value=10,value=3,step=1)
x = st.slider('Select Y of starting point', min_value=0, max_value=10,value=9,step=1)

if rewards[x, y] == -100.:
   st.warning(f'The starting point {x}, {y} is a terminal state. Please select another starting point.')
else:
   if st.button('Calculate Best Path'):
    plot_environment(get_shortest_path(x,y))
