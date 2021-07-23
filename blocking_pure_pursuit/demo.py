import time
import yaml
import gym
import numpy as np
import math
from argparse import Namespace
import examples.mockrospy as rospy

from examples.disparity_extender_head_to_head import DisparityExtenderDriving
from examples.sensor_msgs import LaserScan, drive_params
from pure_pursuit import PurePursuitPlanner
from pure_pursuit import nearest_point_on_trajectory

class LaneSwitcher:
    def __init__(self, lanes, num_lanes, current_lane, proximity_distance=2.5):
        # Assumptions for lanes vector layout: each lane occupies 3 columns (x, y, vel), last three columns is the raceline, previous are lanes from inner to outer, corresponding to increase in indexing
        self.lanes = lanes
        self.num_lanes =  num_lanes
        self.current_lane = current_lane
        self.pd = proximity_distance
        self.switch_thresh = 0.2

    def _rotation_matrix(self, angle, direction, point=None):
        sina = math.sin(angle)
        cosa = math.cos(angle)
        direction = self._unit_vector(direction[:3])
        # rotation matrix around unit vector
        R = np.array(((cosa, 0.0,  0.0),
                      (0.0,  cosa, 0.0),
                      (0.0,  0.0,  cosa)),
                      dtype=np.float64)
        R += np.outer(direction, direction) * (1.0 - cosa)
        direction *= sina
        R += np.array((( 0.0,         -direction[2],  direction[1]),
                       ( direction[2], 0.0,          -direction[0]),
                       (-direction[1], direction[0],  0.0)),
                       dtype=np.float64)
        M = np.identity(4)
        M[:3, :3] = R
        if point is not None:
            # rotation not around origin
            point = np.array(point[:3], dtype=np.float64, copy=False)
            M[:3, 3] = point - np.dot(R, point)
        return M

    def _unit_vector(self, data, axis=None, out=None):
        if out is None:
            data = np.array(data, dtype=np.float64, copy=True)
            if data.ndim == 1:
                data /= math.sqrt(np.dot(data, data))
                return data
        else:
            if out is not data:
                out[:] = np.array(data, copy=False)
            data = out
        length = np.atleast_1d(np.sum(data*data, axis))
        np.sqrt(length, length)
        if axis is not None:
            length = np.expand_dims(length, axis)
        data /= length
        if out is None:
            return data

    def _transform(self, x, y, th, oppx, oppy, oppth):
        """
        Transforms oppponent world coordinates into ego frame

        Args:
            x, y, th (float): ego pose
            oppx, oppy, oppz (float): opponent position

        Returns:
            oppx_new, oppy_new, oppth_new (float): opponent pose in ego frame
        """
        rot = self._rotation_matrix(th, (0, 0, 1))
        homo = np.array([[oppx - x], [oppy - y], [0.], [1.]])
        # inverse transform
        rotated = rot.T @ homo
        rotated = rotated / rotated[3]
        return rotated[0], rotated[1]

    def _pose2quadrant(self, oppx, oppy):
        """
        Returns the quadrant the opponent is in
             x
         1 | 0 | 2
        y----*-----
         3 | 0 | 4

        Args:
            oppx, oppy (float): opponent position in ego frame

        Returns:
            quadrant (int): which area opponent is w.r.t. ego (check figure)
        """
        if oppx >= -0.65:
            if oppy >= 0.1:
                return 1
            elif oppy <= -0.1:
                return 2
            else:
                return 0
        else:
            if oppy >= 1.5:
                return 3
            elif oppy <= -1.5:
                return 4
            else:
                return 0

    def _pose2lane(self, oppx, oppy):
        """
        Returns the lane the opponent is in

        Args:
            oppx, oppy (float): opponent position in world frame

        Returns:
            lane (int): which lane the opponent is on
        """
        nearest_dist = np.inf
        nearest_lane = -1
        for i in range(self.num_lanes):
            lane_test = self.lanes[:, i*3:i*3+2]
            _, test_dist, _, _ = nearest_point_on_trajectory(np.array([oppx, oppy]), lane_test)
            if test_dist < nearest_dist:
                nearest_lane = i
                nearest_dist = test_dist
        return nearest_lane

    def decision(self, x, y, th, oppx, oppy, oppth):
        """
        Makes laneswitch decision, left or right right now

        Args:
            x, y, th (float): ego pose
            oppx, oppy, oppth (float): opponent pose

        Returns:
            move (int): -1, 0, or 1, indicating moving lanes left, stay, or right;
                        2, indicating switching to raceline when no opponent around
                        3, indicating switching to opponent's lane
        """

        # Check if opponent close by
        radius = np.linalg.norm(np.array([x - oppx, y - oppy]))
        if radius <= self.pd:
            # check quadrant in ego frame
            oppx_new, oppy_new = self._transform(x, y, th, oppx, oppy, oppth)
            quadrant = self._pose2quadrant(oppx_new, oppy_new)
            # print('quadrant: ', quadrant)
            # move to inner
            if quadrant == 1:
                return -1 if self.current_lane != 0 else 0
            # move to outer
            elif quadrant == 2:
                return 1 if self.current_lane != self.num_lanes - 1 else 0
            # behind, block according to opponent lane
            else:
                return 3
        else:
            # if opponent not in range, follow race line
            return 2
driving_params = drive_params(0,0)
def driving_params_callback(params):
    global driving_params
    driving_params = params

if __name__ == '__main__':
    # config
    opponent_cripple = 0.9
    with open('config.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    lanes = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
    # assuming 3 columns per lane (x, y, vel)
    num_lanes = int(lanes.shape[1] / 3 - 1)
    # start on race line
    # current_lane = num_lanes + 1
    current_lane = 1
    ego_switcher = LaneSwitcher(lanes, num_lanes, current_lane)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=2)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta],
                                                       [conf.sx2, conf.sy2, conf.stheta2]]))
    env.render()
    planner = PurePursuitPlanner(conf, conf.wheelbase)
    planner.update_waypoints(lanes[:, 3:6])
    opp_planner = DisparityExtenderDriving()

    pub = rospy.Publisher('scan', LaserScan, queue_size=5)
    sub = rospy.Subscriber('drive_parameters', drive_params, driving_params_callback)
    opp_planner.run()
    laptime = 0.0
    start = time.time()

    while not done:
        # lane switch decision
        decision = ego_switcher.decision(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0],
                                         obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1])
        # update current waypoint being followed
        if decision == 2:
            _, d, _, _ = nearest_point_on_trajectory(np.array([obs['poses_x'][0], obs['poses_y'][0]]), lanes[:, -3:-1])
            if d <= ego_switcher.switch_thresh:
                planner.update_waypoints(lanes[:, -3:])
        elif decision == 3:
            opp_lane = ego_switcher._pose2lane(obs['poses_x'][1], obs['poses_y'][1])
            # print('opp_lane', opp_lane)
            planner.update_waypoints(lanes[:, 3*opp_lane:3*opp_lane+3])
        elif decision == -1 and ego_switcher.current_lane != 0:
            ego_switcher.current_lane -= 1
            planner.update_waypoints(lanes[:, 3*ego_switcher.current_lane:3*ego_switcher.current_lane+3])
        elif decision == 1 and ego_switcher.current_lane < ego_switcher.num_lanes:
            ego_switcher.current_lane += 1
            planner.update_waypoints(lanes[:, 3*ego_switcher.current_lane:3*ego_switcher.current_lane+3])
        else:
            pass

        # print('decision', decision, 'current lane', ego_switcher.current_lane)

        speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], conf.lookahead_distance, conf.vgain)
        pub.publish(LaserScan(obs["scans"][1].tolist()))
        # opponent is crippled
        obs, step_reward, done, info = env.step(np.array([[steer, speed],
                                                          [driving_params.angle, driving_params.velocity]]))
        laptime += step_reward
        env.renderer.update_waypoints(planner.waypoints[:, :2])
        env.render(mode='human')
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)