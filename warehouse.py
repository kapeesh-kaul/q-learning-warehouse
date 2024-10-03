import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

class WarehouseEnvironment:
    def __init__(self):
        self.environment_rows = 11
        self.environment_columns = 11
        self.q_values = np.zeros((self.environment_rows, self.environment_columns, 4))
        self.actions = ['up', 'right', 'down', 'left']
        self.rewards = np.full((self.environment_rows, self.environment_columns), -100.)
        self.rewards[0, 5] = 100.  # Set the reward for the packaging area (i.e., the goal) to 100

        # Define aisle locations (i.e., white squares) for rows 1 through 9
        self.aisles = {
            1: [i for i in range(1, 10)],
            2: [1, 9],
            3: [i for i in range(1, 8)] + [9],
            4: [3, 7, 9],
            5: [i for i in range(1, 10)],
            6: [2, 5, 8],
            7: [i for i in range(1, 10)],
            8: [1, 9],
            9: [i for i in range(1, 10)]
        }
        self.set_aisle_rewards()

    def set_aisle_rewards(self):
        for row_index in range(1, 10):
            for column_index in self.aisles[row_index]:
                self.rewards[row_index, column_index] = -1.

    def is_terminal_state(self, current_row_index, current_column_index):
        return self.rewards[current_row_index, current_column_index] != -1.

    def get_starting_location(self):
        current_row_index = np.random.randint(self.environment_rows)
        current_column_index = np.random.randint(self.environment_columns)
        while self.is_terminal_state(current_row_index, current_column_index):
            current_row_index = np.random.randint(self.environment_rows)
            current_column_index = np.random.randint(self.environment_columns)
        return current_row_index, current_column_index

    def get_next_action(self, current_row_index, current_column_index, epsilon):
        if np.random.random() < epsilon:
            return np.argmax(self.q_values[current_row_index, current_column_index])
        else:
            return np.random.randint(4)

    def get_next_location(self, current_row_index, current_column_index, action_index):
        new_row_index, new_column_index = current_row_index, current_column_index
        if self.actions[action_index] == 'up' and current_row_index > 0:
            new_row_index -= 1
        elif self.actions[action_index] == 'right' and current_column_index < self.environment_columns - 1:
            new_column_index += 1
        elif self.actions[action_index] == 'down' and current_row_index < self.environment_rows - 1:
            new_row_index += 1
        elif self.actions[action_index] == 'left' and current_column_index > 0:
            new_column_index -= 1
        return new_row_index, new_column_index

    def get_shortest_path(self, start_row_index, start_column_index):
        if self.is_terminal_state(start_row_index, start_column_index):
            return []
        else:
            current_row_index, current_column_index = start_row_index, start_column_index
            shortest_path = [[current_row_index, current_column_index]]
            while not self.is_terminal_state(current_row_index, current_column_index):
                action_index = self.get_next_action(current_row_index, current_column_index, 1.)
                current_row_index, current_column_index = self.get_next_location(current_row_index, current_column_index, action_index)
                shortest_path.append([current_row_index, current_column_index])
            return shortest_path

    def train(self, epsilon, discount_factor, learning_rate, episodes=1000):
        for episode in range(episodes):
            row_index, column_index = self.get_starting_location()
            while not self.is_terminal_state(row_index, column_index):
                action_index = self.get_next_action(row_index, column_index, epsilon)
                old_row_index, old_column_index = row_index, column_index
                row_index, column_index = self.get_next_location(row_index, column_index, action_index)

                reward = self.rewards[row_index, column_index]
                old_q_value = self.q_values[old_row_index, old_column_index, action_index]
                temporal_difference = reward + (discount_factor * np.max(self.q_values[row_index, column_index])) - old_q_value
                new_q_value = old_q_value + (learning_rate * temporal_difference)
                self.q_values[old_row_index, old_column_index, action_index] = new_q_value
        print('Training complete!')

    def plot_environment(self, path=None):
        fig, ax = plt.subplots(figsize=(2, 2))
        for row in range(self.environment_rows):
            for col in range(self.environment_columns):
                if self.rewards[row, col] == -100.:
                    color = 'black'
                elif self.rewards[row, col] == 100.:
                    color = 'green'
                else:
                    color = 'white'
                ax.add_patch(plt.Rectangle((col, self.environment_rows - row - 1), 1, 1, color=color, edgecolor='gray'))

        if path:
            for (row, col) in path:
                ax.add_patch(plt.Rectangle((col, self.environment_rows - row - 1), 1, 1, color='yellow', edgecolor='gray', alpha=0.5))

        plt.xlim(0, self.environment_columns)
        plt.ylim(0, self.environment_rows)
        ax.set_aspect('equal')
        ax.axis('off')
        st.pyplot(fig, use_container_width=False)