import os
import csv
import matplotlib.pyplot as plt
import numpy as np

def id_to_dir(id):
    return f'experiments/experiment_{id}/data'

def plot_experiment(roots):
    for root in roots:
        max_speeds = []
        for data in os.listdir(id_to_dir(root)):
            state_file = os.path.join(id_to_dir(root), data, 'stateFile.csv')
            if os.path.isfile(state_file):
                with open(state_file, newline='') as state_file_data:
                    reader = csv.DictReader(state_file_data, delimiter=',')

                    max_mario_x = 0
                    for row in reader:
                        curr_mario_x = int(row['MarioX'])
                        max_mario_x = max(max_mario_x, curr_mario_x)
                    max_speeds.append(max_mario_x)
        
        x = max_speeds
        N = 10
        plt.plot(np.convolve(x, np.ones(N)/N, mode='valid'), label=root)

    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_experiment([0, 1, 2])