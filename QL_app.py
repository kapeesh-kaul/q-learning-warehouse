import streamlit as st
from warehouse import WarehouseEnvironment

# Streamlit UI

'''
# CS715: Advanced Machine Learning
## Assignment: Q-Learning 
###  Warehouse Robot Pathfinding
#### Introduction
In this assignment, we have implemented a Q-learning-based solution to optimize the path of a warehouse robot navigating through an environment with various obstacles and goals. The environment is modeled after a warehouse layout, where the robot's task is to find the shortest path from any given starting point to a designated item packaging area.

#### Problem Description
The problem simulates a warehouse environment with multiple states (locations) and obstacles (storage areas). The robot must learn the optimal path to the packaging area using Q-learning. The environment is structured as an 11x11 grid representing different locations:

- Aisles (white squares): Locations the robot can navigate.
- Storage areas (black squares): Obstacles that the robot must avoid.
- Packaging area (green square): The target destination for the robot.
- The goal is to use reinforcement learning (specifically, Q-learning) to help the robot learn the most efficient path to the packaging area while avoiding obstacles.

#### Q-Learning Implementation
The environment consists of states (locations within the grid), actions (movement directions), and rewards:

- States: All possible grid positions within the warehouse, represented as an 11x11 matrix.
- Actions: The robot can move in four possible directions: up, right, down, and left.
- Rewards: The packaging area provides a positive reward (+100), while other navigable locations have a small negative reward (-1) to encourage the robot to find the shortest path. All storage areas have a high negative reward (-100) to discourage the robot from entering them.
'''


st.sidebar.write('## Define Learning Parameters:')
epsilon = st.sidebar.slider('epsilon', min_value=0.3, max_value=1.0, step=0.1, value=0.9)
discount_factor = st.sidebar.slider('discount_factor', min_value=0.3, max_value=1.0, step=0.1, value=0.9)
learning_rate = st.sidebar.slider('learning_rate', min_value=0.3, max_value=1.0, step=0.1, value=0.9)

env = WarehouseEnvironment()
env.train(epsilon, discount_factor, learning_rate)

env.plot_environment()
st.write('## Select Source Coordinates:')
y = st.slider('Select X of starting point', min_value=0, max_value=10, value=3, step=1)
x = st.slider('Select Y of starting point', min_value=0, max_value=10, value=9, step=1)

st.sidebar.info(
    '''
Developed by `kapeesh-kaul` for CS715 to demonstrate Q-Learning in warehouse robot pathfinding. 
    '''
)

if env.rewards[x, y] == -100.:
    st.warning(f'The starting point {x}, {y} is a terminal state. Please select another starting point.')
else:
    if st.button('Calculate Best Path'):
        env.plot_environment(env.get_shortest_path(x, y))
