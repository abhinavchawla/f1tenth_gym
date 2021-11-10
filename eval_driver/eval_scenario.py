"""
eval a specific scenario for overtaking
"""
import pickle
from math import sqrt
from typing import List

import time
import os
from copy import deepcopy
import pyglet

import numpy as np
from scipy import signal

from start_states import pack_odom

from my_laser_models import MyScanSimulator2D

class LidarDrawer:
    """class for drawing lidar data of a car"""

    def __init__(self):

        self.vertex_list = None

        self.x_y_theta_scan = None

        self.color = (0, 255, 0, 32) + (0, 255, 0, 64) #(64, 64, 64)

        self.fov = 4.7

        self.counter = 0
        self.downsample_step = 5

    def render_callback(self, w):
        """render callback, draws lidar data"""

        if self.x_y_theta_scan is not None:
            car_x, car_y, car_theta, scan = self.x_y_theta_scan

            num_pts = 2 * len(scan)

            if self.vertex_list is None:
                pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
                pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
    
                # make initial self.vertex_list

                self.vertex_list = w.batch.add(num_pts, pyglet.gl.GL_LINES, None,
                        ('v3f/stream', np.zeros(3 * num_pts)),
                        ('c4B/static', self.color * len(scan))
                    )

            #pt_list = np.zeros(3 * num_pts) # list of x, y, 0.0 for the lines
            offset = 0

            theta_arr = np.linspace(car_theta - self.fov / 2, car_theta + self.fov / 2, len(scan))

            sines = np.sin(theta_arr)
            cosines = np.cos(theta_arr)

            for dist, sin, cos in zip(scan, sines, cosines):
                x = 50*car_x + 50*dist * cos
                y = 50*car_y + 50*dist * sin

                self.vertex_list.vertices[offset:offset + 6] = (50*car_x, 50*car_y, 0, x, y, 0)
                offset += 6

            #self.vertex_list.vertices = pt_list

            #self.counter += 1
            #theta = self.counter / 50

            #x = 200 * np.cos(theta)
            #y = 200 * np.sin(theta)

            #self.vertex_list.vertices[3:5] = 2*x, y

    def update_pose_lidar(self, x, y, theta, scan):
        """update car pose and lidar data"""

        if self.downsample_step > 1:
            downsampled = []

            for i, d in enumerate(scan):
                if i % self.downsample_step == 0:
                    downsampled.append(d)

            scan = downsampled

        self.x_y_theta_scan = x, y, theta, scan

lidar_drawer = LidarDrawer()
added_render_callback = False


def percent_completed(own_x, own_y):
    'find the percent completed in the race just using the closest waypoint'

    try:
        with open("waypoints_gap.pkl", "rb") as f:
            waypoints = pickle.load(f)
            print(f"loaded {len(waypoints)} waypoints")
    except FileNotFoundError:
        print(f"no cached starting positions found at waypoints, re-running computation")
    min_dist_sq = np.inf
    rv = 0
    num_waypoints =len(waypoints)
    # min_dist_sq is the squared sum of the two legs of a triangle
    # considering two waypoints at a time

    for i in range(len(waypoints) - 2):
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i + 1]

        dx1 = (x1 - own_x)
        dy1 = (y1 - own_y)

        dx2 = (x2 - own_x)
        dy2 = (y2 - own_y)

        dist_sq1 = dx1 * dx1 + dy1 * dy1
        dist_sq2 = dx2 * dx2 + dy2 * dy2
        dist_sq = dist_sq1 + dist_sq2

        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            # 100 * min_index / num_waypoints

            rv = 100 * i / num_waypoints

            # add the fraction completed betwen the waypints
            dist1 = sqrt(dist_sq1)
            dist2 = sqrt(dist_sq2)
            frac = dist1 / (dist1 + dist2)

            assert 0.0 <= frac <= 1.0
            rv += frac / num_waypoints

    return rv

def find_next_opp_state(percent, opp_start_states):
    for i in range(len(opp_start_states)-1):
        p1 = percent_completed(opp_start_states[i][0].state[0], opp_start_states[i][0].state[1])
        p2 = percent_completed(opp_start_states[i+1][0].state[0],opp_start_states[i+1][0].state[1])
        if p1 < percent < p2:
            return i+1
    return None


def eval_scenario(ego_start_tup, opp_start_tup, env, racetrack, opp_start_states, overtake_timeout=60):
    """run an evaluation from the given positions

    returns one of: 'overtake', 'crash', 'overtake_timeout'
    """

    global lidar_drawer
    global added_render_callback

    if not added_render_callback:
        env.add_render_callback(lidar_drawer.render_callback)
        added_render_callback = True

    parent = os.path.dirname(os.path.realpath(__file__))
    map_path = os.path.join(parent, 'maps', f'{racetrack}.png')
    yaml_path = os.path.join(parent, 'maps', f'{racetrack}.yaml')
    scanner = MyScanSimulator2D(map_path, yaml_path)

    ego_racecar, ego_driver = ego_start_tup
    opp_racecar, opp_driver = opp_start_tup
    
    drivers = [deepcopy(ego_driver), deepcopy(opp_driver)]
    start_poses = np.array([
            np.array(ego_racecar),
            np.append(opp_racecar.state[0:2], opp_racecar.state[4])
        ])

    obs, step_reward, done, info = env.reset(poses=start_poses)
    env.render(mode='human_fast')

    # move the vehicles into place
    env.sim.agents[1] = deepcopy(opp_racecar)

    cur_frame = 0
    max_frames = 100 * overtake_timeout
    rv = "overtake_timeout" # gets overwritten
    start = time.perf_counter()
    last_detected_in_back = False
    next_ego_start = None
    next_opp_start = None
    while cur_frame < max_frames:
        cur_frame += 1

        actions = np.zeros((2, 2))

        for i in range(2):
            odom = pack_odom(obs, i)
            scan = obs['scans'][i]

            if i == 0:
                x = odom['pose_x']
                y = odom['pose_y']
                theta = odom['pose_theta']
                ego_pose = np.array([x, y, theta], dtype=float)

                ego_scan = scan
                ego_clean_scan = scanner.scan(ego_pose)

                lidar_drawer.update_pose_lidar(x, y, theta, scan)

            driver = drivers[i]
                
            if hasattr(driver, 'process_observation'):
                speed, steer = driver.process_observation(ranges=scan, ego_odom=odom)
            else:
                assert hasattr(driver, 'process_lidar')
                speed, steer = driver.process_lidar(scan)

            actions[i, 0] = steer
            actions[i, 1] = speed

        obs, step_reward, done, info = env.step(actions)        
        env.render(mode='human_fast')

        if obs['collisions'][0]:
            rv = "crash"
            break;

        assert not done, "Finished race? (done was True)"

        # compare scan with clean scan to see if overtake completed
        # overtake completed if back 1/4 of scan is not clean (opponent is behind) and rest is clean
        f = len(scan) // 4

        back_scan = np.append(ego_scan[-f:], ego_scan[:f])
        back_clean_scan = np.append(ego_clean_scan[-f:], ego_clean_scan[:f])

        max_diff_back = np.max(np.abs(back_scan - back_clean_scan))
        max_diff_front = np.max(np.abs(ego_scan[f:-f] - ego_clean_scan[f:-f]))

        detected_in_front = max_diff_front > 0.1
        detected_in_back = max_diff_back > 0.1

        if last_detected_in_back and not detected_in_back and not detected_in_front:
            # passed opponent
            rv = "overtake"
            next_ego_pose = (env.sim.agents[0].state[0], env.sim.agents[0].state[1], env.sim.agents[0].state[4])
            next_ego_start = (next_ego_pose, ego_driver)
            x = env.sim.agents[0].state[0]
            y = env.sim.agents[0].state[1]
            percent  = percent_completed(x,y)
            next_opp_start = find_next_opp_state(percent, opp_start_states)
            break

        last_detected_in_back = detected_in_back

    diff = time.perf_counter() - start
    print(f'{rv} in {round(diff, 1)} sec')

    return rv, next_ego_start, next_opp_start
