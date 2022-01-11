from scipy.optimize import minimize
import numpy as np
import math
import sys
from matplotlib import pyplot as plt
from astar import *
import cv2
sys.path.append(".")

import cubic_spline_planner

def get_straight_course(dl):
    ax = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    ay = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_straight_course2(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_straight_course3(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    cyaw = [i - math.pi for i in cyaw]

    return cx, cy, cyaw, ck


def get_forward_course(dl):
    ax = [0.0, 60.0, 125.0, 50.0, 75.0, 30.0, -10.0]
    ay = [0.0, 0.0, 50.0, 65.0, 30.0, 50.0, -20.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck

def check_goal(state, goal, tind, nind, stop_speed, goal_dis):

    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.sqrt(dx ** 2 + dy ** 2)

    isgoal = (d <= goal_dis)

    if abs(tind - nind) >= 5:
        isgoal = False

    isstop = (abs(state.v) <= stop_speed)

    if isgoal and isstop:
        return True

    return False

def smooth_yaw(yaw):

    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= np.pi / 2.0:
            yaw[i + 1] -= np.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -np.pi / 2.0:
            yaw[i + 1] += np.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw

def get_switch_back_course(dl):
    ax = [0.0, 30.0, 6.0, 20.0, 35.0]
    ay = [0.0, 0.0, 20.0, 35.0, 20.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    ax = [35.0, 10.0, 0.0, 0.0]
    ay = [20.0, 30.0, 5.0, 0.0]
    cx2, cy2, cyaw2, ck2, s2 = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    cyaw2 = [i - math.pi for i in cyaw2]
    cx.extend(cx2)
    cy.extend(cy2)
    cyaw.extend(cyaw2)
    ck.extend(ck2)
    return cx, cy, cyaw, ck

def get_8_course1(dl):
    ax = [30.0, 27.0 , 20.0, 9.0, 3.0, 0.0, -3.0, -9.0, -20.0, -27.0, -30.0]
    ay = [0.0, -7.0, -10.0, -10.0, -7.0, 0.0, 7.0, 10.0, 10.0, 7.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    ax = [-1*i for i in ax]
    
    cx2, cy2, cyaw2, ck2, s2 = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    cx.extend(cx2)
    cy.extend(cy2)
    cyaw.extend(cyaw2)
    ck.extend(ck2)
    return cx, cy, cyaw, ck

def custom_path(dl):
    area_map = cv2.imread("maps/factory_graph.png")
    src = np.array([1000, 700])
    dest = np.array([110, 454])
    weight = 1
    path, found = a_star(src, dest, area_map, weight)
    ax = list(path[:,0])
    ay = list(path[:,1])
    """
    plt.xlim(-80,80)
    plt.ylim(-80,80)
    plt.plot(ax[:],ay[:])
    plt.show()
    quit()
    """
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    cyaw = [i - math.pi for i in cyaw]

    return cx, cy, cyaw, ck


def pi_2_pi(angle):
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi

    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle

def calc_speed_profile(cx, cy, cyaw, target_speed):

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile

    
def add_motor_noise(v,omega):
    v_  = np.random.normal(v,0.1*np.abs(v))
    omega_  = np.random.normal(omega,0.1*np.abs(omega))
    return v_, omega_