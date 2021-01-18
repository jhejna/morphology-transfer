import os
import numpy as np
import gym
import tempfile
import xml.etree.ElementTree as ET
from .base import GoalEnv

class Reach(GoalEnv):
    TARGET_SIZE = None
    ARENA_SPACE_LOW = None
    ARENA_SPACE_HIGH = None
    TARGET_DIM = None # The current location in ref to target is skill_obs[:TARGET_DIM] = obs[AGENT_DIM:AGENT_DIM+SKILL_DIM][:TARGET_DIM]
    REWARD_SCALE = None
    SPARSE_REWARD = None
    SURVIVE_REWARD = 0
    VISUALIZE = False

    def __init__(self, model_path=None):
        self.target = self.ARENA_SPACE_HIGH # Init to far away from agent start.
        self.num_reached = 0
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "assets", self.ASSET)
        if self.VISUALIZE:
            xml_path = os.path.join(os.path.dirname(__file__), "assets", model_path)
            tree = ET.parse(xml_path)
            world_body = tree.find(".//worldbody")
            _, xml_path = tempfile.mkstemp(text=True, suffix='.xml')
            target_elem = ET.Element('body')
            target_elem.set("name", "target")
            target_elem.set("pos", "0 0 " + str(self.TARGET_SIZE))
            target_geom = ET.SubElement(target_elem, "geom")
            target_geom.set("conaffinity", "0")
            target_geom.set("contype", "0")
            target_geom.set("name", "target")
            target_geom.set("pos", "0 0 0")
            target_geom.set("rgba", "0.2 0.9 0.2 0.8")
            target_geom.set("size", str(self.TARGET_SIZE))
            target_geom.set("type", "sphere")
            world_body.insert(-1, target_elem)
            tree.write(xml_path)
        else:
            xml_path = None
        
        super(Reach, self).__init__(model_path=xml_path)

    def _get_obs(self):
        return NotImplemented

    def compute_reward(self, achieved_goal, desired_goal, info):
        dist_to_target = np.linalg.norm(achieved_goal - desired_goal)
        reward = -1*self.REWARD_SCALE * dist_to_target
        if dist_to_target < self.TARGET_SIZE:
            reward += self.SPARSE_REWARD
        reward += self.SURVIVE_REWARD
        return reward

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        desired_goal = self.target
        achieved_goal = self.skill_obs(obs)[:self.TARGET_DIM]
        reward = self.compute_reward(achieved_goal, desired_goal, None)
        done = False
        if np.linalg.norm(achieved_goal - desired_goal) < self.TARGET_SIZE:
            done = True
        return obs, reward, done, {'success' : done}
        
class ReachNav(Reach):
    ARENA_SPACE_LOW = np.array([-8.0, -8.0])
    ARENA_SPACE_HIGH = np.array([8.0, 8.0])
    SKILL_DIM = 2
    TARGET_DIM = 2
    TASK_DIM = 4 # agent position, target position
    TARGET_SIZE = 0.5
    REWARD_SCALE = 0.1
    SPARSE_REWARD = 50
    VISUALIZE = True

class Reach_PointMass(ReachNav):
    ASSET = 'point_mass.xml'
    AGENT_DIM = 2
    FRAME_SKIP = 3

    def _get_obs(self):
        return {
            'observation' : np.concatenate((self.sim.data.qvel.flat[:],
                                            self.get_body_com("torso")[:2]), axis=0),
            'achieved_goal' : self.get_body_com("torso")[:2],
            'desired_goal' : self.target,
        }

    def reset(self):
        self.target = self.np_random.uniform(low=self.ARENA_SPACE_LOW, high=self.ARENA_SPACE_HIGH)
        if self.VISUALIZE:
            self.model.body_pos[-2][:self.TARGET_DIM] = self.target
        qpos = self.init_qpos + self.np_random.uniform(low=-self.TARGET_SIZE, high=self.TARGET_SIZE, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.2, high=0.2, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

class Reach_Ant(ReachNav):
    ASSET = 'ant.xml'
    AGENT_DIM = 123
    FRAME_SKIP = 5
    
    def _get_obs(self):
        return {
            'observation' : np.concatenate([
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:],
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
                self.get_body_com("torso")[:2]
            ], axis=0),
            'achieved_goal' : self.get_body_com("torso")[:2],
            'desired_goal' : self.target,
        }

    def reset(self):
        self.target = self.np_random.uniform(low=self.ARENA_SPACE_LOW, high=self.ARENA_SPACE_HIGH)
        if self.VISUALIZE:
            self.model.body_pos[-2][:self.TARGET_DIM] = self.target
        qpos = self.np_random.uniform(size=self.model.nq, low=-.1, high=.1) + self.init_qpos
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

class Reach_Quadruped(ReachNav):
    ASSET = 'quadruped.xml'
    AGENT_DIM = 163
    FRAME_SKIP = 5

    def _get_obs(self):
        return {
            'observation' : np.concatenate([
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:],
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
                self.get_body_com("torso")[:2]
            ], axis=0),
            'achieved_goal' : self.get_body_com("torso")[:2],
            'desired_goal' : self.target,
        }

    def reset(self):
        self.target = self.np_random.uniform(low=self.ARENA_SPACE_LOW, high=self.ARENA_SPACE_HIGH)
        if self.VISUALIZE:
            self.model.body_pos[-2][:self.TARGET_DIM] = self.target
        qpos = self.np_random.uniform(size=self.model.nq, low=-.1, high=.1) + self.init_qpos
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()
    

class ReachSawyer(Reach):
    FRAME_SKIP = 1
    NSUBSTEPS = 20
    ARENA_SPACE_LOW = np.array([0.5, -0.2, -0.1])
    ARENA_SPACE_HIGH = np.array([1.1, 0.2, 0.5])
    SKILL_DIM = 3
    TARGET_DIM = 3
    TASK_DIM = 6 # peg end position, goal position, but could include more (grip base, peg_top, etc.)
    TARGET_SIZE = 0.025
    REWARD_SCALE = 0.1
    SPARSE_REWARD = 10
    SURVIVE_REWARD = -0.5
    VISUALIZE = True
    JOINT_RANDOMIZATION_LOW = None
    JOINT_RANDOMIZATION_HIGH = None
    NUM_JOINTS = None

    def __init__(self, relative_grip_pos=True, rand_init=False, valid_area=False):
        self.relative_grip_pos = relative_grip_pos
        self.rand_init = rand_init
        self.valid_area = valid_area
        # Remove the table for the reach task.
        xml_path = os.path.join(os.path.dirname(__file__), "assets", self.ASSET)
        tree = ET.parse(xml_path)
        world_body_elem = tree.find(".//worldbody")            
        for body in world_body_elem.findall('body'):
            if body.get('name') == 'table':
                world_body_elem.remove(body)
        _, xml_path = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(xml_path)

        # Add fake control bounds for init
        self.control_low = -100*np.ones(self.NUM_JOINTS)
        self.control_high = 100*np.ones(self.NUM_JOINTS)

        super(ReachSawyer, self).__init__(model_path=xml_path)
        # Now we have the initialized model. Change the action space etc to be scaled correctly.
        bounds = self.sim.model.actuator_ctrlrange[:self.NUM_JOINTS]
        self.control_low = np.copy(bounds[:, 0])
        self.control_high = np.copy(bounds[:, 1])
        self.action_space = gym.spaces.Box(low=-1*np.ones(self.NUM_JOINTS), high=np.ones(self.NUM_JOINTS), dtype=np.float32)
    
    def scale_action(self, action):
        act_k = (self.control_high - self.control_low) / 2.
        act_b = (self.control_high + self.control_low) / 2.
        return act_k * action + act_b

    def get_qpos(self):
        angle_noise_range = 0.015
        qpos = self.sim.data.qpos[:self.NUM_JOINTS]
        qpos += np.random.uniform(-angle_noise_range,
                                  angle_noise_range,
                                  self.NUM_JOINTS)
        
        return qpos.reshape(-1)

    def get_qvel(self):
        velocity_noise_range = 0.015
        qvel = self.sim.data.qvel[:self.NUM_JOINTS]
        qvel += np.random.uniform(-velocity_noise_range,
                                  velocity_noise_range,
                                  self.NUM_JOINTS)
        return qvel.reshape(-1)

    def _get_obs(self):
        ref_pt = self.get_site_com('ref_pt')
        peg_top = self.get_site_com('peg_top') - ref_pt
        grip_base = self.get_site_com('grip_base')
        if self.relative_grip_pos:
            grip_base -= ref_pt
        return {
            'observation' : np.concatenate([
                self.get_qpos(),
                self.get_qvel(),
                grip_base,
                peg_top,
                ref_pt[:self.TARGET_DIM]
            ], axis=0),
            'achieved_goal' : ref_pt[:self.TARGET_DIM],
            'desired_goal' : self.target,
        }
    
    def in_valid_area(self):
        x_ref, y_ref, z_ref = self.get_site_com('ref_pt')
        if x_ref < 0.15 or abs(y_ref) > 0.5 or z_ref > 0.9 or z_ref < -0.25:
            return False
        else:
            return True

    def step(self, action):
        action = self.scale_action(action)
        obs, reward, done, info = super(ReachSawyer, self).step(action)
        # print(obs, self.target)
        if self.valid_area and not self.in_valid_area():
            done = True
        return obs, reward, done, info

    def reset(self):
        self.sim.reset()
        self.target = self.np_random.uniform(low=self.ARENA_SPACE_LOW, high=self.ARENA_SPACE_HIGH)
        if self.VISUALIZE:
            self.model.body_pos[-2][:self.TARGET_DIM] = self.target
        if self.rand_init:
            qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=self.JOINT_RANDOMIZATION_LOW, high=self.JOINT_RANDOMIZATION_HIGH)
            qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .05
            self.set_state(qpos, qvel)
            while not self.in_valid_area():
                qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=self.JOINT_RANDOMIZATION_LOW, high=self.JOINT_RANDOMIZATION_HIGH)
                qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .05
                self.set_state(qpos, qvel)
        else:
            qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.05, high=.05)
            qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .05
            self.set_state(qpos, qvel)
        return self._get_obs()

class Reach_Sawyer5Arm1(ReachSawyer):
    JOINT_RANDOMIZATION_LOW = [-0.5, -1.25, -1.0, -1.0, -0.5]
    JOINT_RANDOMIZATION_HIGH = [0.5, 0.25, 1.75, 1.0, 0.5]
    ASSET = 'sawyer/5dof_robot1_peg.xml'
    AGENT_DIM = 5*2 + 6
    NUM_JOINTS = 5
