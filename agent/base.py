from abc import ABC, abstractmethod
from collections import deque
import torch
import numpy as np

class BaseAgent(ABC):
    def __init__(self, config):
        self.config = config
        
    def save(self, file_name):
        torch.save(self.config.network.state_dict(), f'{file_name}.pth')

    def load(self, file_name):
        # the map_location call transfers the storage already to the correct device
        state_dict = torch.load(f'{file_name}.pth', map_location=lambda storage, loc: storage)
        self.config.network.load_state_dict(state_dict)

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def save(self, file_name):
        pass

    @abstractmethod
    def load(self, file_name):
        pass

    def train(self):
        '''
            Train an agent for a given number of episodes
        '''
        config = self.config
        scores_deque = deque(maxlen=config.reward_window_size)
        scores = []
        for i_episode in range(1, config.num_episodes+1):
            self.reset()
            episide_score = 0
            for _ in range(config.max_steps):
                score, done = self.step()
                episide_score += score
                if done:
                    break 

            scores_deque.append(episide_score)
            scores.append(episide_score)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_deque)), end="")
            
            if i_episode % config.save_interval == 0:
                self.save('checkpoint.pth', scores)

            if i_episode % config.log_interval == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                    i_episode, np.mean(scores_deque)))

        return scores

    def run(self):
        '''
            Run an agent for one episode without training
        '''
        total_score = 0
        for _ in range(self.config.max_steps):
            score, done = self.step(skip_training=True)
            total_score += score
            if done:
                break
        print(f'Total score {total_score}')