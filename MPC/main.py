import numpy as np
#import pytomlpp as  toml
import toml
import time
from matplotlib import pyplot as plt
from path_helper import *
from mpc import MPC, State
#from nonideal_motor import Motor

config = toml.load('config.toml')


ref_courses = {
        1: get_straight_course(config['dl']),
        2: get_straight_course2(config['dl']),
        3: get_straight_course3(config['dl']),
        4: get_switch_back_course(config['dl']),
        5: get_8_course1(config['dl']),
        6: custom_path(config['dl'])
        }

#get reference path
cx, cy, cyaw, ck = ref_courses[ config['ref_course'] ]
goal = [cx[-1], cy[-1]]

sp = calc_speed_profile(cx, cy, cyaw, config['target_speed'])
t = int(config['target_speed']/(config['min_acceleration']*config['dt']))
for i in range(len(sp)-t,len(sp)-1):
    sp[i] = sp[i-1]+config['min_acceleration']*config['dt']

#put all references into a single matrix
ref = np.array([cx,cy,sp,cyaw])
ref[3] = smooth_yaw(ref[3])
print(ref[2])

#set initial state
state = State(x=ref[0][0], y=ref[1][0], yaw=ref[3][0], v=0.0)


#lists to store evolving states
x_list = []
y_list = []
cv_list = []
v_list = []
yaw_list = []
omega_list = [0]
cte_list = []

ov,oomega = (None,None)
ind=0
i = 0

#initialize time variables
try:
    if(config['time_it']):
        total_time = time.time()
        total_time -= total_time
except:
    config['time_it'] = False

print('x,y,v,yaw,next_v,omega',end='')

if(config['time_it']):
    print(',time',end='')
print()

# motor noise/ non-ideal motor
add_noise = config['motor_noise']

if add_noise==2:
    drive_motor = Motor(**config["drive_motor_params"])
    steering_motor = Motor(**config["steering_motor_params"])

#initialize MPC class with config in TOML file
mympc = MPC('config.toml')

for i in range(config['n_mpc']):
    start = time.time()

    # storing the states
    x_list.append(state.x)
    y_list.append(state.y)
    cv_list.append(state.v)
    yaw_list.append(state.yaw)
    cte_list.append(mympc.calc_cte(state,ref,ind))

    #calling the mpc solver
    ov,oomega,ind = mympc.solve(state,ref,ov,oomega,ind)

    #select only the first step
    v,omega = ov[0], oomega[0]


    #add motor noise
    if add_noise==1:
        v, omega = add_motor_noise(v, omega)
    elif add_noise==2:
        for _ in range(config['n_motor']):
            drive_motor.update_response(v)
            steering_motor.update_response(omega)
        v, omega = drive_motor.state, steering_motor.state

    omega_list.append(np.rad2deg(omega))
    v_list.append(v)

    print(state.x, state.y, state.v, np.rad2deg(state.yaw), v, np.rad2deg(omega), sep=',', end='')
    #update states
    state.x = state.x + state.v * np.cos(state.yaw) * mympc.dt
    state.y = state.y + state.v * np.sin(state.yaw) * mympc.dt
    state.v = v
    state.omega = omega
    if config['model'] == 'bicycle':
        state.yaw = state.yaw + state.v * np.tan(omega) * mympc.dt
    elif config['model'] == 'unicycle':
        state.yaw = state.yaw + omega

    #check if goal state is reached
    isgoal = check_goal(state, goal, ind, len(cx), config['stop_speed'], config['goal_dis'])

    #calculate time taken
    if(config['time_it']):
        end = time.time()
        curr_time = end-start
        total_time+=curr_time
        print('',curr_time,sep=',',end='')
    print()

    #stop if goal is reached
    if isgoal:
        print("goal !!")
        break

if config['time_it']:
    print('Average time per MPC.solve call:', total_time/i)

# plotting
title = 'target_speed:'+str(config['target_speed'])+" ; "+'max_v:' + str(config['max_speed']) +" ; " + 'max_a:'+ str(config['max_acceleration'])+" ; " + 'max_steer:' + str(config['max_steer'])

fig1, ax1 = plt.subplots()
ax1.set_title(title)
ax1.plot(cx,cy, color='blue', label='reference path')
ax1.plot(x_list, y_list, color='red', label='tracked path')
ax1.legend(loc='upper right')
fig1.savefig(config['tracked_plot_name'])

fig2, ax2 = plt.subplots()
ax2.set_title(title)
ax2.plot(v_list, color = 'blue', label='velocity')
ax2.plot(omega_list, color = 'red', label='omega/steer')
ax2.plot(cte_list, color = 'green', label='CTE')
ax2.legend(loc='upper right')
fig2.savefig(config['v_omega_cte_plot_name'])

plt.show()