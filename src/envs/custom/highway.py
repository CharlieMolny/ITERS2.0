from highway_env import utils
import random
import numpy as np
import copy
from highway_env.envs import highway_env
from highway_env.vehicle.controller import ControlledVehicle
import torch
from src.feedback.feedback_processing import encode_trajectory


class CustomHighwayEnv(highway_env.HighwayEnvFast):

    def __init__(self, shaping=False, time_window=5,run_tailgaiting=False):
        super().__init__()
        self.shaping = shaping
        self.time_window = time_window
        self.run_tailgaiting=run_tailgaiting

        self.episode = []
        self.number_of_features=5
        #number of states= 5 , number_of features
        #self.state_len = 5 ##need to change this
        if self.run_tailgaiting:
            self.number_of_states=5
            
        else:
            self.number_of_states =1
        self.state_len = self.number_of_features*self.number_of_states  
        self.lows = np.zeros((self.state_len, ))
        self.highs = np.ones((self.state_len, ))

        # speed is in [-1, 1]
        self.lows[[3, 4]] = -1
        self.action_dtype = 'int'
        self.state_dtype = 'cont'

        self.lane = 0
        self.lmbda = 0.2
        self.epsilon=0
        self.lane_changed = []

        if  self.run_tailgaiting:
            self.config['tailgating'] = {}
            self.config['tailgating']['threshold'] = 0
            self.config['tailgating']['reward'] = 0


        # presence features are immutable
        self.immutable_features = [0]

        self.discrete_features = [0]
        self.cont_features = [f for f in range(self.state_len) if f not in self.discrete_features]

        self.max_changed_lanes = 3

    def step(self, action):
        self.episode.append((self.state, action))

        curr_lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) else self.vehicle.lane_index[2]
        self.state, rew, done, info = super().step(action) ## super is the original highway-- (inheritence)
                                                        
        info['true_rew'] = rew

        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        self.lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) else self.vehicle.lane_index[2]

        tailgating=False
        if self.run_tailgaiting:
            for i,other_vehicle in enumerate(self.road.vehicles[1:10]): ## must get rid of index 0, the closest vehicless appear first in the list 
                other_vehicle_lane=other_vehicle.target_lane_index[2] if isinstance(self.road.vehicles, ControlledVehicle) else self.road.vehicles[1].target_lane_index[2]
                if other_vehicle_lane !=self.lane or other_vehicle.crashed:
                    continue
                distance_between_vehicles=abs(other_vehicle.position[0]-self.vehicle.position[0]) 
                threshold=self.config['tailgating']['threshold']
                if distance_between_vehicles < threshold:
                    tailgating =True
            
            tailgating_rew=0.0
            if tailgating:
                tailgating_rew=self.config['tailgating']['reward']
            

        
        self.lane_changed.append(self.lane != curr_lane)

        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])

        coll_rew = self.config["collision_reward"] * self.vehicle.crashed
        right_lane_rew = self.config["right_lane_reward"] * self.lane / max(len(neighbours) - 1, 1)
        speed_rew = self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)

        lane_change = sum(self.lane_changed[-self.time_window:]) >= self.max_changed_lanes
        true_reward = self.calculate_true_reward(rew, lane_change,tailgating)

        aug_rew = 0
        if self.shaping:
            aug_rew = self.augment_reward(action, self.state)

        rew += aug_rew
        lane_change_weight=self.config['lane_change_reward']
        add_lane_change=lane_change * lane_change_weight
        rew += add_lane_change

        if self.run_tailgaiting:
            rew += tailgating_rew    
        
            info['rewards'] = {'collision_rew': coll_rew,
                            'right_lane_rew': right_lane_rew,
                            'speed_rew': speed_rew,
                            'tailgating_rew': tailgating_rew,
                            'lane_change_rew': aug_rew,   
                            'lane_changed': lane_change,
                            'true_reward': true_reward}
        else: 
            info['rewards'] = {'collision_rew': coll_rew,
                'right_lane_rew': right_lane_rew,
                'speed_rew': speed_rew,
                'lane_change_rew': aug_rew,   
                'lane_changed': lane_change,
                'true_reward': true_reward}

        return self.state, rew, done, info

    def calculate_true_reward(self, rew, lane_change,tailgating):
        tailgating_rew=0
        if tailgating:
            tailgating_rew=self.true_rewards['tailgating']['reward']

        true_rew = rew + tailgating_rew + self.true_rewards['lane_change_reward'] * lane_change 
        return true_rew

    def reset(self):
        self.episode = []
        self.lane_changed = []
        self.state= super().reset()

        self.lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) else self.vehicle.lane_index[2]

        return self.state

    def close(self):
        pass

    def render(self):
        super().render(mode='human')

    def render_state(self, state):
        print('State = {}'.format(state.flatten()[0:5]))

    def augment_reward(self, action, state):
        running_rew = 0
        past = copy.copy(self.episode)
        curr = 1

        for j in range(len(past)-1, -1, -1):  # go backwards in the past
            state_enc = encode_trajectory(past[j:], state, curr, self.time_window, self)

            rew = self.reward_model.predict(state_enc)
            running_rew += self.lmbda * rew.item()

            if curr >= self.time_window:
                break

            curr += 1

        return running_rew

    def set_reward_model(self, rm):
        self.reward_model = rm

    def set_shaping(self, boolean):
        self.shaping = boolean

    def set_true_reward(self, rewards):
        self.true_rewards = rewards

    def random_state(self):
        return np.random.uniform(self.lows, self.highs, (self.state_len, ))

    def encode_state(self, state):
        if self.run_tailgaiting:
            encoded_state=state.flatten() ### original
        else:
            encoded_state=state[0].flatten() 
        return encoded_state
          
    def valid_feature_indices(self,rule):
        return  [index + 1 for index, (feature, details) in enumerate(rule['features'].items()) if details['Expression'] is not None]


    def get_tailgaiting_feedback(self, best_traj, expl_type):
        feedback_list=[]
        for traj in best_traj:

            threshold=1
            sensitivity=0.1
            states= [s for s,a in traj]

            very_negative_signal=-4
            feedback_added = False
            
            for i,state in enumerate(states):
                # Skip the rest of the trajectory if feedback has already been added
                if feedback_added:
                    break
                        
                ego_vehicle_x, ego_vehicle_y = state[0][1], state[0][2]
                for location in state[1:]:
                    other_vehicle_x, other_vehicle_y=location[1],location[2]
                    if abs(other_vehicle_y-ego_vehicle_y) >0.1: ### vehicles are in different lanes
                        continue
                    if abs(other_vehicle_x-ego_vehicle_x) < threshold:
                        print("Very Negative Trajectory")

                        rule = {
                                'quant': 's',
                                'features': {
                                    'Feature1': {
                                        'Expression': {
                                            'type': '-',  
                                            'abs': True, 
                                            'threshold': threshold  
                                        }
                                    },
                                    'Feature2': {
                                        'Expression': '==',
                                        'sensitivity': sensitivity

                                    },
                                    'Feature3': {
                                        'Expression': None  
                                    },
                                    'Feature4': {
                                        'Expression': None  
                                    }
                                }
                            }   
                        start=max(0, i - (self.time_window//2))
                        end=min(len(traj), i +  (self.time_window//2) + 1)
                        valid_feature_indices=self.valid_feature_indices(rule)


                        total_length =self.number_of_states*(end-start)  # Total length up to which we want to mark indices

                        # Calculate the number of states needed to cover the total length
                        num_states_needed = total_length // (self.state_len // self.number_of_features)

                        important_features = [
                            index + (i * (self.state_len // self.number_of_features))
                            for i in range(num_states_needed)
                            for index in valid_feature_indices
                        ]


                        feedback=('s', traj[ start:end], very_negative_signal,important_features,rule,self.time_window) 
                        feedback_list.append(feedback)
                        feedback_added = True
                        break
        return feedback_list, True

   

 ##  new get binary feedback
    def get_lane_feedback(self, best_traj, expl_type):
        feedback_list = []
        count = 0
        for traj in best_traj:

            # Extract lane positions from each state in the trajectory
            lanes = [s.flatten()[2] for s, a in traj]
            # Determine if a lane change occurred between consecutive states
            changed_lanes = [abs(lanes[i] - lanes[i-1]) > 0.1 if i >= 1 else False for i, _ in enumerate(lanes)]

            start = 0
            end = start + 2
            negative_label_assigned = False  

            while end < len(changed_lanes):
                while (end - start) <= self.time_window:
                    if end >= len(changed_lanes):
                        break

                    changed = sum(changed_lanes[(start+1):end]) >= self.max_changed_lanes

                    if changed and changed_lanes[start+1]:  
                        print("Negative Trajectory number {}".format(count))
                        negative_label_assigned = True  # Set flag to True as negative label is assigned
                        signal=-1
                        if random.random() < self.epsilon:
                               signal=-signal
                        feedback_list.append(('s', traj[start:end], signal, [2 + (i*self.state_len) for i in range(0, end-start)], {},end-start))
                        start = end  # Move to the next segment
                        end = start + 2
                        if expl_type == 'expl':
                            break
                    else:
                        end += 1  # Expand the current segment

                start += 1  # Move the start forward for the next evaluation
                end = start + 2  # Reset the end of the segment

            # If no negative label was assigned throughout the trajectory, label it positive
            if not negative_label_assigned:
                print("Positive Trajectory Number {}".format(count))
                lowerbound, upperbound = 0, len(traj)
                if len(traj) > self.time_window:
                    # If the trajectory is longer than the time_window, select a random segment
                    start = random.randint(0, len(traj) - self.time_window)  # Random start index
                    lowerbound, upperbound = start, start + self.time_window
                    signal=1
                    if random.random() < self.epsilon:
                        signal=-signal
                feedback_list.append(('s', traj[lowerbound:upperbound], signal, [2 + (i*self.state_len) for i in range(0, upperbound-lowerbound)],{},upperbound-lowerbound ))

            count += 1
        return feedback_list, True

    def get_feedback(self, best_traj, expl_type):
        
        #feedback,_= self.get_tailgaiting_feedback( best_traj, expl_type)
        feedback,_= self.get_lane_feedback( best_traj, expl_type)
        return feedback, True

    def set_lambda(self, l):
        self.lmbda = l
    
    def set_epsilon(self, e):
        self.epsilon=e




