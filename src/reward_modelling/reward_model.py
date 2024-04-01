import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from src.reward_modelling.replay_buffer import ReplayBuffer
from src.reward_modelling.reward_nn import RewardModelNN




class RewardModel:

    def __init__(self, time_window, input_size,max_human_rew=9999,lmbda=0,load_iteration=False):
        self.time_window = time_window
        self.buffer = ReplayBuffer(capacity=10000, time_window=self.time_window)
        self.predictor = RewardModelNN(input_size)
        

        if load_iteration:

            self.buffer.dataset=torch.load(r'datasets\highway_tailgating\buffer\seed_0_lmbda_0.1_epsilon_1data.pt') ##chanage this later
            state_dict=torch.load(r'reward_models\highway_tailgaiting\seed_0_lmbda_0.1_epsilon_1')
            self.predictor.net.load_state_dict(state_dict)
            print("loaded previous iteration")
        


        if max_human_rew is not None:
            self.buffer.set_maximum_marked(lmbda=lmbda, maximum_human_rew=max_human_rew)

    def update(self):
        dataset = self.buffer.get_dataset()
        train, test = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))])
        print("length training datset {}".format(len(train)))
        self.predictor.train(DataLoader(train, shuffle=True, batch_size=512))
        self.predictor.evaluate(DataLoader(test, shuffle=True, batch_size=512))

    def update_buffer(self, D, signal, important_features, datatype, actions, rules, iter):
        self.buffer.update(D, signal, important_features, datatype, actions, rules, iter)

    def predict(self, encoding):
        encoding = np.array(encoding).reshape(1, -1)
        return self.predictor.predict(encoding)
    
    def save(self, file_path='model.pth'):

        # Ensure the directory exists
        directory = os.path.dirname(file_path)
        if directory != '' and not os.path.exists(directory):
            os.makedirs(directory)

        # Save the model
        torch.save(self.predictor.net.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def get_buffer(self):
        return self.buffer.get_dataset()
        




