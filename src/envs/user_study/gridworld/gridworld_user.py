import pygame
import sys
import numpy as np
import random
import time 

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
GREEN = (0, 200, 0)  
YELLOW = (200, 200, 0)  
RED = (200, 0, 0)  
ORANGE = (255, 165, 0)

feedback_list=[]
running_feedback={}

sidebar_width = 250 
screen_width, screen_height = 700, 600 
button_width, button_height = 100, 40
button_spacing = 10
button_margin = 20
grid_area_width = screen_width - sidebar_width  
grid_size = 5
cell_size = grid_area_width // grid_size

done_button_pos = (screen_width - sidebar_width + (sidebar_width - button_width) / 2, 30)
done_button_text = "Done"
done_visible = False  

repeat_button_pos = (screen_width - sidebar_width + (sidebar_width - button_width) / 2, 100)  
repeat_button_size = (button_width, button_height)
repeat_button_rect = pygame.Rect(repeat_button_pos, repeat_button_size)
repeat_button_color = (0, 255, 0)  
repeat_button_text = "Repeat"
repeat_button_text_color = (255, 255, 255) 

button_positions = [
    (grid_area_width + (sidebar_width - 2 * button_width - button_spacing) / 2, 
    screen_height - 3 * (button_height + button_margin)),
    (grid_area_width + (sidebar_width - 2 * button_width - button_spacing) / 2 + button_width + button_spacing, 
    screen_height - 3 * (button_height + button_margin)),
    (grid_area_width + (sidebar_width - 2 * button_width - button_spacing) / 2, 
    screen_height - 2 * (button_height + button_margin)),
    (grid_area_width + (sidebar_width - 2 * button_width - button_spacing) / 2 + button_width + button_spacing, 
    screen_height - 2 * (button_height + button_margin)),     
    (grid_area_width + (sidebar_width - button_width) / 2, 
    screen_height - (button_height + button_margin)),
]

signal_button_texts = ["Positive", "Negative", "Very Negative","None"]
feedback_type_button_texts = ["Action", "State", "Rule","None"]
state_button_texts=["Pirate X","Pirate Y","Treasure X","Treasure Y","Orientation"]
button_states = [False] * len(state_button_texts)
current_buttons, current_state = signal_button_texts, 'signal'
current_buttons = signal_button_texts
current_traj_index = 0
none_state=False
action_state=False

pirate_pos = [0, 0]  
pirate_orientation = 0  
treasure_pos = [0, 0]  
current_traj_index = 0  

def rotate_pirate_anticlockwise():
    global pirate_orientation
    pirate_orientation=(pirate_orientation + 1) % 4  # Rotate anticlockwise

def move_pirate():
    global pirate_pos
    if pirate_orientation== 0 and  pirate_pos[0]+1<grid_size:
            pirate_pos[0] +=1
    elif pirate_orientation == 1 and pirate_pos[1]+1 < grid_size:
            pirate_pos[1] += 1
    elif pirate_orientation == 2 and pirate_pos[0]-1 >=0 :
            pirate_pos[0] -= 1
    elif pirate_orientation == 3  and pirate_pos[1]-1 >= 0:
            pirate_pos[1] -= 1

def draw_buttons(button_texts,screen):
    button_font = pygame.font.SysFont("Arial", 18)
    for i, (x, y) in enumerate(button_positions):
        if i < len(button_texts): 
            if button_texts[i]=="Positive":
                color=GREEN
            elif button_texts[i]=="Negative":
                color=YELLOW
            elif button_texts[i]=="Very Negative":
                color=RED
            elif button_texts[i]=="None":
                color =ORANGE if none_state else GREY
            elif button_texts[i]=="Action":
                color =ORANGE if action_state else GREY
            elif button_texts[i]=="Pirate X":
                color = ORANGE if button_states[i] else GREY
            elif button_texts[i]=="Pirate Y":
                color = ORANGE if button_states[i] else GREY
            elif button_texts[i]=="Treasure X":
                color = ORANGE if button_states[i] else GREY
            elif button_texts[i]=="Treasure Y":
                color = ORANGE if button_states[i] else GREY
            elif button_texts[i]=="Orientation":
                color = ORANGE if button_states[i] else GREY
            else:
                color=GREY
            pygame.draw.rect(screen, color, [x, y, button_width, button_height])
            text = button_font.render(button_texts[i], True, BLACK)
            screen.blit(text, text.get_rect(center=(x + button_width / 2, y + button_height / 2)))

    if not done_visible:
        pygame.draw.rect(screen, WHITE, (*done_button_pos, button_width, button_height))

    if done_visible:
        pygame.draw.rect(screen, GREY, (*done_button_pos, button_width, button_height))
        done_text_surf = button_font.render(done_button_text, True, BLACK)
        screen.blit(done_text_surf, done_text_surf.get_rect(center=(done_button_pos[0] + button_width / 2, done_button_pos[1] + button_height / 2)))

    pygame.draw.rect(screen, repeat_button_color, (*repeat_button_pos,button_width,button_height))
    repeat_text_surf=button_font.render("Repeat",True,BLACK)
    screen.blit(repeat_text_surf, repeat_text_surf.get_rect(center=(repeat_button_pos[0] + button_width / 2, repeat_button_pos[1] + button_height / 2)))


def initialise_trajectory(best_traj,start_index):
        global current_traj_index
        global pirate_pos
        global treasure_pos
        global pirate_orientation
        global none_state,done_visible
        global button_states, action_state
        global current_state,current_buttons

        initial_state = best_traj[current_traj_index][start_index][0]
        init_agent_x, init_agent_y, init_goal_x, init_goal_y, init_orientation = initial_state
        pirate_pos = [init_agent_x, init_agent_y]
        treasure_pos = [init_goal_x, init_goal_y]
        pirate_orientation = init_orientation
        none_state=False
        done_visible=False
        action_state=False
        button_states = [False] * len(state_button_texts)
        current_buttons, current_state = signal_button_texts, 'signal'

def check_button_click(pos,best_traj,traj,start_index):
    for i, (x, y) in enumerate(button_positions):
        if x <= pos[0] <= x + button_width and y <= pos[1] <= y + button_height:
            handle_button_click(i)
            return "Cont"
    if done_visible and done_button_pos[0] <= pos[0] <= done_button_pos[0] + button_width and done_button_pos[1] <= pos[1] <= done_button_pos[1] + button_height:
        global current_traj_index
        current_traj_index += 1  

        if current_traj_index<len(best_traj):
            running_feedback['traj']=traj[start_index:start_index+4]
            running_feedback['start index']=start_index
            initialise_trajectory(best_traj,start_index)
            print(running_feedback)
            feedback_list.append(running_feedback)
        return "Done"
    if repeat_button_pos[0] <= pos[0] <= repeat_button_pos[0] + button_width and repeat_button_pos[1] <= pos[1] <= repeat_button_pos[1] + button_height:
        initialise_trajectory(best_traj,start_index)
        return "Repeat"

def handle_button_click(index):
    global current_buttons, current_state
    global done_visible, none_state,action_state,running_feedback
    print(f"{current_buttons[index]} clicked")
    if current_state == 'signal':
        if current_buttons[index]=="None":
            none_state = not none_state
            done_visible=not done_visible
            running_feedback={}
        else:
            running_feedback['signal']=current_buttons[index]
            current_buttons, current_state = feedback_type_button_texts, 'feedback'
    elif  current_buttons[index]=='Action':
        done_visible =not done_visible
        action_state=not action_state
        if action_state: ##action has been pressed
            running_feedback["feedback_type"]='a'
        else:
            running_feedback={}
            
    elif current_state == 'feedback':
        current_buttons, current_state = state_button_texts, 'state'
    elif current_state == 'state':
        button_states[index] = not button_states[index]
        done_visible=True
        running_feedback["feedback_type"]='s'
        for idx, state in enumerate(button_states):
            if state:  
                running_feedback[current_buttons[idx]] = state

def play_trajectory(traj_index,start_index,traj,screen,pirate_img,treasure_img):
        last_update = pygame.time.get_ticks()
        update_delay = 1000
        while  traj_index < start_index + 4:

            current_time = pygame.time.get_ticks()
            if current_time - last_update > update_delay:
                last_update = current_time
                state, action = traj[traj_index]
                agent_x, agent_y, goal_x, goal_y, orientation = state

                assert(agent_x == pirate_pos[0] and agent_y == pirate_pos[1])
                assert(goal_x == treasure_pos[0] and goal_y == treasure_pos[1])
                assert(orientation == pirate_orientation)

                if action == 1:
                    rotate_pirate_anticlockwise()
                elif action == 0:
                    move_pirate()

                traj_index += 1
                screen.fill(WHITE)
                for x in range(0, grid_area_width, cell_size):
                    for y in range(0, grid_area_width, cell_size):
                        pygame.draw.rect(screen, BLUE, pygame.Rect(x, y, cell_size, cell_size), 1)

                rotation_angle = pirate_orientation * -90
                rotated_pirate_img = pygame.transform.rotate(pirate_img, rotation_angle)
                screen.blit(rotated_pirate_img, (pirate_pos[0] * cell_size, pirate_pos[1] * cell_size))
                screen.blit(treasure_img, (treasure_pos[0] * cell_size, treasure_pos[1] * cell_size))
                
                pygame.display.flip()

def process_trajectory(best_traj, traj, screen, pirate_img, treasure_img):
    assert len(traj) > 4
    start_index = random.randint(0, len(traj) - 4)
    traj_index = start_index

    initial_state = traj[start_index][0]
    init_agent_x, init_agent_y, init_goal_x, init_goal_y, init_orientation = initial_state
    global pirate_pos, pirate_orientation, treasure_pos
    pirate_pos = [init_agent_x, init_agent_y]
    treasure_pos = [init_goal_x, init_goal_y]
    pirate_orientation = init_orientation

    play_trajectory(traj_index,start_index,traj,screen,pirate_img,treasure_img)

    done = False
    while not done:
        draw_buttons(current_buttons,screen)  
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True  
            elif event.type == pygame.MOUSEBUTTONDOWN:
                terminal_button=check_button_click(event.pos,best_traj,traj,start_index)
                if terminal_button=="Done":
                    
                    done = True
                elif terminal_button=="Repeat":
                    play_trajectory(traj_index,start_index,traj,screen,pirate_img,treasure_img)      
        pygame.display.flip()
        if current_traj_index >= len(best_traj):
            pygame.quit()
     
def convert_feedback():
    feature_indices = {
    'Pirate X': 0,
    'Pirate Y': 1,
    'Treasure X': 2,
    'Treasure Y':3,
    'Orientation':4
}
    feedback_tuple_list=[]
    for feedback_dict in feedback_list:
        feedback_type=feedback_dict['feedback_type']
        feedback_traj = feedback_dict['traj']
        signal=feedback_dict['signal']
        important_features = [feature_indices[key] for key in feedback_dict if key in feature_indices and feedback_dict[key]]
        feedback_entry = (feedback_type, feedback_traj, signal, important_features, {}, 4)
        feedback_tuple_list.append(feedback_entry)
    return feedback_tuple_list

def run_user_study(best_traj):
    pygame.init()
    global directions
    directions = ['left', 'down', 'right', 'up']

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Pirate Treasure Hunt')

    pirate_img = pygame.transform.scale(pygame.image.load(r'src\envs\user_study\gridworld\pirate.png'), (cell_size, cell_size))
    treasure_img = pygame.transform.scale(pygame.image.load(r'src\envs\user_study\gridworld\treasure.png'), (cell_size, cell_size))

    for traj in best_traj:
        process_trajectory(best_traj,traj,screen,pirate_img,treasure_img)
        time.sleep(2)

    feedback_tuple_list= convert_feedback()
    return feedback_tuple_list

    



