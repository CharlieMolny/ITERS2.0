import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import csv
import os
from src.feedback.feedback_processing import satisfy


def check_environment():
    # Check if the script is running in Google Colab
    if 'COLAB_GPU' in os.environ:
        return False
    else:
        return True 


class ReplayBuffer:

    def __init__(self, capacity, time_window):
        self.capacity = capacity
        self.time_window = time_window
        self.count=0
        self.curr_iter = 0
        self.maximum_mark=999999999 ### setting to arbitrarily large

    def initialize(self, dataset,load_iteration):
        self.dataset = dataset

        # measures how many times a trajectory was added
        self.marked = np.zeros((len(self.dataset), ))
        if load_iteration:
            self.marked = self.dataset.tensors[1]



    def update_original(self, new_data, signal, important_features, datatype, actions, rules, iter): #self, new_data, signal, important_features, datatype, actions, rules, iter
        print('Updating reward buffer...')
        full_dataset = torch.cat([self.dataset.tensors[0], new_data.tensors[0]])
        curr_dataset = self.dataset

        y = torch.cat([curr_dataset.tensors[1], new_data.tensors[1]])
        y = [signal if self.similar_to_data(new_data.tensors[0], full_dataset[i], important_features, datatype, actions, rules) else np.sign(l) for i, l in enumerate(y)]
        y = torch.tensor(y)

        threshold = 0.05

        if self.curr_iter != iter:
            closest = [self.closest(n, self.dataset.tensors[0], important_features, rules) for n in new_data.tensors[0]]
            new_marked = [max(self.marked[closest[i][0]]) + 1 if closest[i][1] < threshold else 1 for i, n in enumerate(new_data.tensors[0])]
            new_marked = torch.tensor(new_marked)

            self.marked = [m + 1 if self.similar_to_data(new_data.tensors[0], self.dataset.tensors[0][i], important_features, datatype, actions, rules) else m for i, m in enumerate(self.marked)]
            self.marked = torch.tensor(self.marked)
            self.marked = torch.cat([self.marked, new_marked])

            y = self.marked * y
        else:
            closest = [self.closest(n, self.dataset.tensors[0], important_features, rules) for n in new_data.tensors[0]]
            new_marked = [max(self.marked[closest[i][0]]) if closest[i][1] < threshold else 1 for i, n in
                          enumerate(new_data.tensors[0])]
            new_marked = torch.tensor(new_marked)

            self.marked = [m if self.similar_to_data(new_data.tensors[0], self.dataset.tensors[0][i], important_features,
                                              datatype, actions, rules) else m for i, m in enumerate(self.marked)]
            self.marked = torch.tensor(self.marked)
            self.marked = torch.cat([self.marked, new_marked])

            y = self.marked * y

        self.dataset = TensorDataset(full_dataset, y)
        self.curr_iter = iter

    def  update(self, new_data, signal, important_features, datatype, actions, rules, iter):
         
        print("Original Dataset shape: {}".format(self.dataset.tensors[0].shape))
        print("New Dataset shape : {}".format(new_data.tensors[0].shape))
        
        full_dataset = torch.cat([self.dataset.tensors[0], new_data.tensors[0]])
        curr_dataset = self.dataset
        # print("size of full dataset {}".format(len(full_dataset)))

        threshold = 0.05
        closest = [self.closest(n, self.dataset.tensors[0], important_features, rules) for n in new_data.tensors[0]]

        new_marked = [max(self.marked[closest[i][0]],key=abs) + signal if closest[i][1] < threshold and max(self.marked[closest[i][0]],key=abs) + signal < self.maximum_mark+ signal else signal for i, n in enumerate(new_data.tensors[0])]
            
        new_marked = torch.tensor(new_marked)

        #new_marked = torch.clamp(new_marked, min=None, max=self.maximum_mark)
        new_marked_list=new_marked.tolist()

        updated_marked = []

        for i, current_mark in enumerate(self.marked):
            new_data_tensor = new_data.tensors[0]
            dataset_tensor = self.dataset.tensors[0][i]

            is_similar = self.similar_to_data(
                new_data_tensor, 
                dataset_tensor, 
                important_features, 
                datatype, 
                actions, 
                rules
            )
            
            if is_similar and current_mark+signal<self.maximum_mark:
                updated_mark = current_mark + signal
            else:
                updated_mark = current_mark
            

            updated_marked.append(updated_mark)

        self.marked = updated_marked

        print("Maximum Signal Allowed in Buffer {}".format(self.maximum_mark))
        self.marked = torch.tensor(self.marked)
        self.marked = torch.cat([self.marked, new_marked])
        marked_list = self.marked.tolist()
        min_marked_value = min(marked_list)
        max_marked_value = max(marked_list)

        prefix=''

        filename = prefix+"minmax.csv"

        # Writing to the CSV file
        with open(filename, 'a', newline='') as csvfile:
            # Create a CSV writer object
            
            csvwriter = csv.writer(csvfile)
            
            csvwriter.writerow([self.curr_iter,min_marked_value, max_marked_value])

            print("Minimum signal in Buffer: ", min_marked_value)
            print("Maximum signal in Buffer: ", max_marked_value)

        self.marked=torch.clamp(self.marked,max=self.maximum_mark)
        
        self.dataset = TensorDataset(full_dataset, self.marked)
        self.curr_iter = iter


    def similar_to_data(self, data, x, important_features, datatype, actions, rules, threshold=0.05):
        if len(rules):
            similar, _ = satisfy(np.array(x.unsqueeze(0)), rules[0], self.time_window)
            return len(similar) > 0

        state_dtype, action_dtype = datatype
        if (state_dtype == 'int' and not actions) or (action_dtype == 'int' and actions):
            im_feature_vals = x[important_features]
            exists = torch.where((data[:, important_features] == im_feature_vals).all())
            return len(exists[0]) > 0
        elif (state_dtype == 'cont' and not actions) or (action_dtype == 'cont' and actions):
            mean_features = torch.mean(data, axis=0)
            similarity = abs(mean_features[important_features] - x[important_features])
            returnVaribale=(similarity < threshold).all().item()
            return returnVaribale

    def closest(self, x, data, important_features, rules):
        if len(rules):
            if rules['quant']=='a':
                close_data, close_indices = satisfy(np.array(data), rules[0], self.time_window)
            elif rules['quant']=='s':  
                 close_data, close_indices = satisfy(np.array(data), rules, self.time_window)
            return close_indices, np.zeros((len(close_indices), ))

        difference = torch.mean(abs(data[:, important_features] - x[important_features]) * 1.0, axis=1)
        min_indices = [torch.argmin(difference, dim=-1).item()]

        # min_indices = torch.where(difference == min_diff)[0]
        returnDifference = difference[min_indices[0]].item()
        return min_indices, returnDifference

    def get_data_loader(self,):
        return DataLoader(self.dataset, batch_size=256, shuffle=True)

    def get_dataset(self):
        print('Unique values in labels = {}'.format(torch.unique(self.dataset.tensors[1], return_counts=True)))
        return self.dataset
   
    def set_maximum_marked(self,lmbda,maximum_human_rew):
        maximum_mark=maximum_human_rew/(lmbda*19)  ### needs to be scaled down to avoid exploding human reward
        self.maximum_mark=maximum_mark