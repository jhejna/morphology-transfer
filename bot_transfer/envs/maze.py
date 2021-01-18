import os
import numpy as np
import tempfile
import gym
import xml.etree.ElementTree as ET
from .base import Env
import math

def construct_maze(maze_id=0, length=1):
    # define the maze to use
    if maze_id == 0:
        if length != 1:
            raise NotImplementedError("Maze_id 0 only has length 1!")
        structure = [
            [1, 1, 1, 1, 1],
            [1, 'r', 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 'g', 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
        return structure
    else:
        raise NotImplementedError("The provided MazeId is not recognized")

class Maze(Env):
    VISUALIZE = True
    SCALING = 8.0
    MAZE_ID = 0
    DIST_REWARD = 0
    SPARSE_REWARD = 100.0
    RANDOM_GOALS = False
    HEIGHT = 2

    # Fixed constants for agents
    SKILL_DIM = 2 # X, Y
    TASK_DIM = 4 # agent position, goal position.

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "assets", self.ASSET)
        
        # Initialize the maze and its parameters
        self.STRUCTURE = construct_maze(maze_id=self.MAZE_ID, length=1)
        torso_x, torso_y = self.get_agent_start()
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y

        tree = ET.parse(model_path)
        worldbody = tree.find(".//worldbody") 
        for i in range(len(self.STRUCTURE)):
            for j in range(len(self.STRUCTURE[0])):
                if self.STRUCTURE[i][j] == 'g':
                    self.current_goal_pos = (i,j)
                if (isinstance(self.STRUCTURE[i][j], int) or isinstance(self.STRUCTURE[i][j], float)) \
                    and self.STRUCTURE[i][j] > 0:
                    height = float(self.STRUCTURE[i][j])
                    ET.SubElement(
                        worldbody, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="%f %f %f" % (j * self.SCALING - torso_x,
                                        i * self.SCALING - torso_y,
                                        self.HEIGHT / 2 * height),
                        size="%f %f %f" % (0.5 * self.SCALING,
                                        0.5 * self.SCALING,
                                        self.HEIGHT / 2 * height),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="%f %f 0.3 1" % (height * 0.3, height * 0.3)
                    )

        if self.VISUALIZE:
            world_body = tree.find(".//worldbody")
            waypoint_elem = ET.Element('body')
            waypoint_elem.set("name", "waypoint")
            waypoint_elem.set("pos", "0 0 " + str(self.SCALING/10))
            waypoint_geom = ET.SubElement(waypoint_elem, "geom")
            waypoint_geom.set("conaffinity", "0")
            waypoint_geom.set("contype", "0")
            waypoint_geom.set("name", "waypoint")
            waypoint_geom.set("pos", "0 0 0")
            waypoint_geom.set("rgba", "0.2 0.9 0.2 0.8")
            waypoint_geom.set("size", str(self.SCALING/10))
            waypoint_geom.set("type", "sphere")
            world_body.insert(-1, waypoint_elem)
            xml_path = model_path

        _, xml_path = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(xml_path)
        
        # Get the list of possible segments of the maze to be the goal.
        self.possible_goal_positions = list()
        for i in range(len(self.STRUCTURE)):
            for j in range(len(self.STRUCTURE[0])):
                if self.STRUCTURE[i][j] == 0 or self.STRUCTURE[i][j] == 'g':
                    self.possible_goal_positions.append((i,j))
        self.goal_range = self.get_goal_range()
        self.center_goal = np.array([(self.goal_range[0] + self.goal_range[1]) / 2, 
                                     (self.goal_range[2] + self.goal_range[3]) / 2])

        super(Maze, self).__init__(model_path=xml_path)

    def sample_goal_pos(self):
        if not self.RANDOM_GOALS:
            return
        cur_x, cur_y = self.current_goal_pos
        self.STRUCTURE[cur_x][cur_y] = 0
        new_x, new_y = self.possible_goal_positions[self.np_random.randint(low=0, high=len(self.possible_goal_positions))]
        self.STRUCTURE[new_x][new_y] = 'g'
        self.current_goal_pos = (new_x, new_y)
        self.goal_range = self.get_goal_range()
        self.center_goal = np.array([(self.goal_range[0] + self.goal_range[1]) / 2, 
                                     (self.goal_range[2] + self.goal_range[3]) / 2])

    def get_agent_start(self):
        for i in range(len(self.STRUCTURE)):
            for j in range(len(self.STRUCTURE[0])):
                if self.STRUCTURE[i][j] == 'r':
                    return j * self.SCALING, i * self.SCALING
        assert False

    def get_goal_range(self):
        for i in range(len(self.STRUCTURE)):
            for j in range(len(self.STRUCTURE[0])):
                if self.STRUCTURE[i][j] == 'g':
                    minx = j * self.SCALING - self.SCALING * 0.5 - self._init_torso_x
                    maxx = j * self.SCALING + self.SCALING * 0.5 - self._init_torso_x
                    miny = i * self.SCALING - self.SCALING * 0.5 - self._init_torso_y
                    maxy = i * self.SCALING + self.SCALING * 0.5 - self._init_torso_y
                    return minx, maxx, miny, maxy

    def _get_obs(self):
        return NotImplemented

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        # Compute the reward
        minx, maxx, miny, maxy = self.goal_range
        x, y = self.get_body_com("torso")[:2]
        reward = 0
        if minx <= x <= maxx and miny <= y <= maxy:
            reward += self.SPARSE_REWARD
            done = True
        else:
            done = False
        if self.DIST_REWARD > 0:
            # adds L2 reward
            reward += -self.DIST_REWARD * np.linalg.norm(self.skill_obs(obs)[:2] - self.center_goal)
        
        return obs, reward, done, {'is_success' : done}

    def reset(self):
        return NotImplemented

class MazeEnd_PointMass(Maze):
    ASSET = 'point_mass.xml'
    AGENT_DIM = 2
    FRAME_SKIP = 3

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qvel.flat[:],
            self.get_body_com("torso")[:2],
            self.center_goal,
        ])
    
    def reset(self):
        self.sim.reset()
        self.sample_goal_pos()
        if self.VISUALIZE:
            self.model.body_pos[-2][:2] = self.center_goal
        qpos = self.init_qpos + self.np_random.uniform(low=-self.SCALE/10.0, high=self.SCALE/10.0, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.2, high=0.2, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

class MazeSample_PointMass(MazeEnd_PointMass):
    RANDOM_GOALS = True

class MazeEnd_Ant(Maze):
    ASSET = 'ant.xml'
    AGENT_DIM = 123
    FRAME_SKIP = 5
    
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:],
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.get_body_com("torso")[:2],
            self.center_goal,
        ])

    def step(self, action):
        obs, reward, done, info = super(MazeEnd_Ant, self).step(action)
        # if not np.isfinite(obs).all() or obs[0] > 1.0 or obs[0] < 0.2:
        #     done = True
        return obs, reward, done, info 

    def reset(self):
        self.sim.reset()
        self.sample_goal_pos()
        if self.VISUALIZE:
            self.model.body_pos[-2][:2] = self.center_goal
        qpos = self.np_random.uniform(size=self.model.nq, low=-.1, high=.1) + self.init_qpos
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

class MazeSample_Ant(MazeEnd_Ant):
    RANDOM_GOALS = True

class MazeEnd_Quadruped(Maze):
    ASSET = 'quadruped.xml'
    AGENT_DIM = 163
    FRAME_SKIP = 5
    
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:],
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.get_body_com("torso")[:2],
            self.center_goal,
        ])

    def step(self, action):
        obs, reward, done, info = super(MazeEnd_Quadruped, self).step(action)
        # TODO: Determine correct height check for quadruped.
        # if not np.isfinite(obs).all() or obs[2] > 1.0 or obs[2] < 0.2:
        #     done = True
        return obs, reward, done, info 

    def reset(self):
        self.sim.reset()
        self.sample_goal_pos()
        if self.VISUALIZE:
            self.model.body_pos[-2][:2] = self.center_goal
        qpos = self.np_random.uniform(size=self.model.nq, low=-.1, high=.1) + self.init_qpos
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

class MazeSample_Quadruped(MazeEnd_Quadruped):
    RANDOM_GOALS = True
