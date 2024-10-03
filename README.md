# Q-Learning Warehouse Robot Pathfinding

## Overview

This application demonstrates the use of Q-Learning, a reinforcement learning technique, to optimize the path of a warehouse robot navigating through a simulated warehouse environment. The robot learns to find the shortest path to a designated packaging area while avoiding obstacles, based on rewards and penalties.

The app is built with Python and Streamlit, providing an interactive interface to visualize the robot's learning process and pathfinding results.

## Features

- **Simulated Warehouse Environment**: An 11x11 grid representing aisles, storage areas (obstacles), and a packaging area (goal).
- **Q-Learning Implementation**: The robot navigates the warehouse environment, learning the optimal path using Q-Learning.
- **Interactive Parameters**: Allows users to adjust learning parameters (epsilon, discount factor, and learning rate) through a Streamlit sidebar.
- **Visual Pathfinding**: Visualizes the warehouse layout and the robot's shortest path to the packaging area.
- **User Input**: Lets users select the starting coordinates of the robot to find the optimal path.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/kapeesh-kaul/q-learning-warehouse.git
    ```
2. Navigate to the project directory:
    ```bash
    cd q-learning-warehouse-app
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the application:
    ```bash
    streamlit run QL_app.py
    ```

## Usage

- After launching the application, use the sidebar to set the Q-learning parameters (epsilon, discount factor, learning rate).
- Visualize the warehouse environment and adjust the starting coordinates for the robot.
- Click on "Calculate Best Path" to see the robot's optimal path from the selected starting point to the packaging area.
- Observe how different parameters affect the robot's pathfinding efficiency.

## Project Structure

- `QL_app.py`: Contains the Streamlit interface for user interaction, visualization, and parameter adjustment.
- `warehouse.py`: Implements the warehouse environment and the Q-Learning algorithm.
- `requirements.txt`: Lists all necessary Python libraries to run the application.

## Requirements

- Python 3.x
- Streamlit
- Numpy
- Matplotlib

## About This App

Developed by Kapeesh Kaul for the CS715: Advanced Machine Learning course to demonstrate Q-Learning in warehouse robot pathfinding.

## License

This project is licensed under the MIT License.

## Acknowledgements

- CS715: Advanced Machine Learning course for inspiring this project.
- Streamlit for providing an easy-to-use interface for building and sharing data applications.
