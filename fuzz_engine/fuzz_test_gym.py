'''
Interface for fuzz tester using gym environment
'''

import time
import yaml
import numpy as np
from math import sqrt

from argparse import Namespace
from f110_gym.envs import F110Env
from matplotlib import pyplot as plt

from FrenetPlanner_Multi_Vehicle import FrenetPlaner, FrenetControllers, PurePursuitPlanner
from cps_fuzz_tester import SimulationState, run_fuzz_testing, calculate_coverage
from smooth_blocking_vs_blocking import LaneSwitcherPlanner

class GapFollower:
    BUBBLE_RADIUS = 160 # was 260
    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 80
    MAX_LIDAR_DIST = 3000000
    STRAIGHTS_SPEED = 8.0 # orig: 8.0
    CORNERS_SPEED = 5.0
    STRAIGHTS_STEERING_ANGLE = np.pi / 18 # orig: np.pi / 18  # 10 degrees

    def __init__(self):
        # used when calculating the angles of the LiDAR data
        self.radians_per_elem = None

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        self.radians_per_elem = (2 * np.pi) / len(ranges)
        # we won't use the LiDAR data from directly behind us
        proc_ranges = np.array(ranges[135:-135])
        # sets each value to the mean over a given window
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
            free_space_ranges: list of LiDAR data which contains a 'bubble' of zeros
        """
        # mask the bubble
        masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)
        # get a slice for each contigous sequence of non-bubble data
        slices = np.ma.notmasked_contiguous(masked)
        max_len = slices[0].stop - slices[0].start
        chosen_slice = slices[0]
        # I think we will only ever have a maximum of 2 slices but will handle an
        # indefinitely sized list for portablility
        for sl in slices[1:]:
            sl_len = sl.stop - sl.start
            if sl_len > max_len:
                max_len = sl_len
                chosen_slice = sl
        return chosen_slice.start, chosen_slice.stop

    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indices of max-gap range, respectively
        Return index of best point in ranges
        Naive: Choose the furthest point within ranges and go there
        """
        # do a sliding window average over the data in the max gap, this will
        # help the car to avoid hitting corners
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE),
                                       'same') / self.BEST_POINT_CONV_SIZE
        return averaged_max_gap.argmax() + start_i

    def get_angle(self, range_index, range_len):
        """ Get the angle of a particular element in the LiDAR data and transform it into an appropriate steering angle
        """
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2
        return steering_angle

    def process_lidar(self, ranges):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        proc_ranges = self.preprocess_lidar(ranges)
        # Find closest point to LiDAR
        closest = proc_ranges.argmin()

        # Eliminate all points inside 'bubble' (set them to zero)
        min_index = closest - self.BUBBLE_RADIUS
        max_index = closest + self.BUBBLE_RADIUS
        if min_index < 0: min_index = 0
        if max_index >= len(proc_ranges): max_index = len(proc_ranges) - 1
        proc_ranges[min_index:max_index] = 0

        # Find max length gap
        gap_start, gap_end = self.find_max_gap(proc_ranges)

        # Find the best point in the gap
        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        # Publish Drive message
        steering_angle = self.get_angle(best, len(proc_ranges))
        if abs(steering_angle) > self.STRAIGHTS_STEERING_ANGLE:
            speed = self.CORNERS_SPEED
        else:
            speed = self.STRAIGHTS_SPEED
        #print('Steering angle in degrees: {}'.format((steering_angle / (np.pi / 2)) * 90))
        return speed, steering_angle

def render_callback(env_renderer):
    'custom extra drawing function'

    e = env_renderer

    # update camera to follow car
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.score_label.x = left
    e.score_label.y = top - 700
    e.left = left - 800
    e.right = right + 800
    e.top = top + 800
    e.bottom = bottom - 800

class F110GymSim(SimulationState):
    'simulation state for fuzzing'

    render_on = True

    @staticmethod
    def get_cmds():
        'get a list of commands (strings) that can be passed into step_sim'

        return ['opp_normal', 'opp_slower']

    @staticmethod
    def get_obs_data():
        '''get labels and ranges on observations

        returns:
            list of 3-tuples, label, min, max
        '''

        return ('Ego Completed Percent', 0, 100), ('Opponent Behind Percent', -10, 10)
        
    def __init__(self):
        # config
        self.work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.25}
        with open('config_Spielberg_map.yaml') as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        conf = Namespace(**conf_dict)


        #env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=2)
        env = F110Env(map=conf.map_path, map_ext=conf.map_ext, num_agents=2)

        # env.add_render_callback(render_callback)

        obs, _step_reward, done, _info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta],
                                                           [conf.sx2, conf.sy2, conf.stheta2]]))

        lanes = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=1)
        center_lane_index = 1
        self.center_lane = lanes[:, center_lane_index*3:center_lane_index*3+2]

        #env.render()

        ego_planner = FrenetPlaner(conf, env, 0.17145 + 0.15875)
        ego_controller = FrenetControllers(conf, 0.17145 + 0.15875)
        opp_planner = PurePursuitPlanner(conf, 0.17145 + 0.15875)

        # do first action here since we have obs
        # ego_lidar = obs['scans'][0]
        # opp_lidar = obs['scans'][1]
        # ego_pose = obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]
        # opp_pose = obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1]
        # ego_planner.update(ego_pose, opp_pose)
        # opp_planner.update(opp_pose, ego_pose)

        # print('decision', decision, 'current lane', ego_switcher.current_lane)
        path = ego_planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0],
                            obs['poses_x'][1], obs['poses_y'][1])

        speed, steer = ego_planner.control(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0], path, ego_controller)
        opp_speed, opp_steer = opp_planner.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], self.work['tlad'],
                                       self.work['vgain'])
        # env.add_render_callback(ego_planner.render_waypoints)
        # env.add_render_callback(opp_planner.render_waypoints)
        self.error = done
        self.num_steps = 0
        self.next_cmds = [[steer, speed], [opp_steer, opp_speed]]
        self.env = env
        self.ego_planner = ego_planner
        self.opp_planner = opp_planner
        self.ego_controller = ego_controller

    def render(self):
        'display visualization'

        self.env.render(mode='human_fast')
        time.sleep(0.1)
            
    def step_sim(self, cmd):
        'step the simulation state'

        assert not self.error
        control_count = 15

        for i in range(106):
            obs, _step_reward, done, _info = self.env.step(np.array(self.next_cmds))

            if F110GymSim.render_on:
                self.env.render(mode='human_fast')

            if done:
                print(f"crashed on step {self.num_steps} substep {i}")
                self.error = True # crashed!
                break

            # ego_lidar = obs['scans'][0]
            # opp_lidar = obs['scans'][1]
            # ego_pose = obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]
            # opp_pose = obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1]
            #
            # self.ego_planner.update(ego_pose, opp_pose)
            # self.opp_planner.update(opp_pose, ego_pose)

            # speed, steer = self.ego_planner.process_lidar(ego_lidar)
            # opp_speed, opp_steer = self.opp_planner.process_lidar(opp_lidar)
            if control_count == 15:
                path = self.ego_planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0],
                                    obs['linear_vels_x'][0], obs['poses_x'][1], obs['poses_y'][1])
                control_count = 0

            speed, steer = self.ego_planner.control(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0],
                                           obs['linear_vels_x'][0], path, self.ego_controller)
            opp_speed, opp_steer = self.opp_planner.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], self.work['tlad'],
                                           self.work['vgain'])
            control_count = control_count + 1

            if cmd == 'opp_slower':
                opp_speed *= 0.8

            self.next_cmds = [[steer, speed], [opp_steer, opp_speed]]

        self.num_steps += 1

    def get_status(self):
        "get simulation status. element of ['ok', 'stop', 'error']"

        ego_x, _opp_x = self.env.render_obs['poses_x']
        ego_y, _opp_y = self.env.render_obs['poses_y']
        ego_percent = self.percent_completed(ego_x, ego_y)

        if self.error:
            rv = 'error'
        elif ego_percent > 95:
            rv = 'stop'
        else:
            rv = 'ok'

        return rv

    def get_obs(self):
        '''get observation of current state

        currently this is a pair, [perent_completed_ego, dist_opp_percent]
        '''

        ego_x, opp_x = self.env.render_obs['poses_x']
        ego_y, opp_y = self.env.render_obs['poses_y']

        ego_percent = self.percent_completed(ego_x, ego_y)
        opp_percent = self.percent_completed(opp_x, opp_y)

        opp_behind_percent = ego_percent - opp_percent

        return np.array([ego_percent, opp_behind_percent], dtype=float)

    def percent_completed(self, own_x, own_y):
        'find the percent completed in the race just using the closest waypoint'

        num_waypoints = self.center_lane.shape[0]

        min_dist_sq = np.inf
        rv = 0

        # min_dist_sq is the squared sum of the two legs of a triangle
        # considering two waypoints at a time

        for i in range(len(self.center_lane) - 1):
            x1, y1 = self.center_lane[i]
            x2, y2 = self.center_lane[i+1]

            dx1 = (x1 - own_x)
            dy1 = (y1 - own_y)

            dx2 = (x2 - own_x)
            dy2 = (y2 - own_y)
            
            dist_sq1 = dx1*dx1 + dy1*dy1
            dist_sq2 = dx2*dx2 + dy2*dy2
            dist_sq = dist_sq1 + dist_sq2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                # 100 * min_index / num_waypoints
                
                rv = 100 * i / num_waypoints

                # add the fraction completed between the waypoints
                dist1 = sqrt(dist_sq1)
                dist2 = sqrt(dist_sq2)
                frac = dist1 / (dist1 + dist2)

                assert 0.0 <= frac <= 1.0
                rv += frac / num_waypoints

        return rv

def main():
    'main entry point'

    F110GymSim.render_on = True

    prefix_filename_1 = 'root_rrt'
    prefix_filename_2 = 'root_random'

    for i in range(1):
        run_fuzz_testing(F110GymSim, seed=i, always_from_start=False, filename=prefix_filename_1+str(i)+'.pkl')
        run_fuzz_testing(F110GymSim, seed=i, always_from_start=True, filename=prefix_filename_2+str(i)+'.pkl')

    for i in range(1):
        calculate_coverage(prefix_filename_1 + str(i)+'.pkl')
        calculate_coverage(prefix_filename_2 + str(i)+'.pkl')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
