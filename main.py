from stable_baselines3 import DQN
from src.envs.custom.gridworld import Gridworld
from src.envs.custom.highway import CustomHighwayEnv
from src.envs.custom.inventory import Inventory
from src.evaluation.evaluator import Evaluator
from src.feedback.rule_feedback import give_rule_feedback
from src.tasks.task import Task
import csv
from src.tasks.task_util import train_expert_model, train_model
from src.util import seed_everything, load_config
import argparse
import torch, os, pickle
from src.visualization.visualization import visualize_experiments, visualize_best_experiment, \
    visualize_best_vs_rand_summary
import os

def check_environment():
    # Check if the script is running in Google Colab
    if 'COLAB_GPU' in os.environ:
        return False
    else:
        return True 


def run(task_name,debugging,prefix):

    run_tailgaiting=False
    run_speed=False 
    # print('Task = {}'.format(task_name))
    rt=''
    rs=''
    if run_tailgaiting:
        rt='_tailgating'

    elif run_speed:
        rs='_speed'
    
    
    # Define paths
    model_path = prefix+'trained_models/{}{}'.format(task_name,rt)

    env_config_path =  prefix+'config/env/{}{}.json'.format(task_name,rt,rs)
    model_config_path =  prefix+'config/model/{}.json'.format(task_name)
    task_config_path =  prefix+'config/task/{}.json'.format(task_name)
    dataset_path= prefix+'datasets/{}{}/'.format(task_name,rt)

    # Load configs
    env_config = load_config(env_config_path)
    model_config = load_config(model_config_path)
    task_config = load_config(task_config_path)

    if task_name == 'gridworld':
        env = Gridworld(env_config['time_window'], shaping=False)
    elif task_name == 'highway':
        env = CustomHighwayEnv(shaping=False, time_window=env_config['time_window'],run_tailgaiting=run_tailgaiting,run_speed=run_speed)
        env.config['right_lane_reward'] = env_config['right_lane_reward']
        env.config['lanes_count'] = env_config['lanes_count']
        env.reset()
    elif task_name == 'inventory':
        env = Inventory(time_window=env_config['time_window'], shaping=False)

    # set true reward function
    env.set_true_reward(env_config['true_reward_func'])


    max_iter = 20 ### changed this so would  finally save

    # initialize starting and expert.csv model

    
    init_model_path =  prefix+'trained_models/{}{}_init'.format(task_name,rs)
    expert_path = prefix+ 'trained_models/{}_expert{}'.format(task_name,rt)
    eval_path = prefix+ 'eval/{}{}{}/'.format(task_name,rt,rs)


       
    model_env = train_model(env, model_config, init_model_path, eval_path, task_config['feedback_freq'], max_iter,debugging)
    expert_model = train_expert_model(env, env_config, model_config, expert_path, eval_path, task_config['feedback_freq'], max_iter, debugging)
    

    seeds = [0]
    lmbdas = [.09]   ##
    epsilons=[1]
    # evaluate experiments
    experiments = [('best_summary', 'expl'), ('best_summary', 'no_exp'), ('rand_summary', 'expl')]

        
    for sum, expl in experiments:   
        for e in epsilons:
            for l in lmbdas:
                for s in seeds:
                        print('Running experiment with summary = {}, expl = {}, lambda = {}, seed = {}, epsilon = {}'.format(sum, expl, l, s,e))
                        seed_everything(s)

                        eval_path = 'eval/{}/{}_{}/'.format(task_name, sum, expl)+rt

                        task = Task(env, model_path,dataset_path, model_env, expert_model, task_name, max_iter, env_config, model_config,
                                    eval_path, debugging,**task_config, expl_type=expl, auto=True, seed=s,run_tailgating=run_tailgaiting,run_speed=run_speed,lmbda=l,prefix=prefix)
                        task.run(experiment_type='regular', lmbda=l, summary_type=sum, expl_type=expl,epsilon=e,prefix=prefix)



def evaluate(task_name,prefix):
        # # visualizing true reward for different values of lambda
    eval_path =prefix +   'eval/{}/best_summary_expl/IRS.csv'.format(task_name)
    original_eval=prefix+'eval/{}/best_summary_expl/IRS_original.csv'.format(task_name)
    best_summary_path = prefix + eval_path
    rand_summary_path = prefix + 'eval/{}/rand_summary_expl/IRS.csv'.format(task_name)
    expert_path = prefix + 'eval/{}/expert'.format(task_name)
    model_env_path = prefix + 'eval/{}/model_env.csv'.format(task_name)
    

    
    eval_paths=[[eval_path,'WITERS'],[original_eval,'ITERS']]

    for eval,title in eval_paths:
        title = '{} for different values of \u03BB in {} task'.format(title,task_name)

        visualize_best_experiment(eval, expert_path, model_env_path, task_name, title)    



    # visualize_best_experiment(eval_path, expert_path, model_env_path, task_name, 'ITERS for different values of \u03BB in Inventory Management task')
    #visualize_best_vs_rand_summary(best_summary_path, rand_summary_path, lmbdas, task_name, 'ITERS for different summary types in GridWorld task')

    # model_path_A = 'trained_models/{}/regular_best_summary_expl/seed_0_lmbda_0.1_iter_5'.format(task_name)
    # model_path_B = 'trained_models/{}/regular_best_summary_expl/seed_0_lmbda_0.2_iter_6'.format(task_name)
    #
    # model_A = DQN.load(model_path_A, env=env)
    # model_B = DQN.load(model_path_B, env=env)
    #
    # give_rule_feedback(model_A, model_B, env)


def main():
    # parser = argparse.ArgumentParser()  
    # parser.add_argument('--task') 
    # args = parser.parse_args()
    # ## add whether it is sumulated feedback here
    # task_name = args.task
    
    debugging= True
    if debugging:
        print("!Debugging!")
    
    local=check_environment()
    if local:
        prefix=''
    else :
        prefix='/content/ITERS2.0/'

    task_name="highway"
    #task_name="gridworld"
    #task_name="inventory"    
    run(task_name,debugging,prefix)
    #evaluate(task_name,prefix)

   


if __name__ == '__main__':
    main()