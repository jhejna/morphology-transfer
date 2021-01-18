import os
import numpy as np
import tempfile
import gym
import xml.etree.ElementTree as ET
from .base import GoalEnv, Env

class PegInsert(Env):
    EPSILON = None
    REWARD_SCALE = None
    SPARSE_REWARD = None

    def __init__(self, model_path=None):
        super(PegInsert, self).__init__()

    def _get_obs(self):
        return NotImplemented

    def _get_peg_obs(self, skill_obs):
        return NotImplemented

    def _get_target_pos(self):
        return NotImplemented

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        ref_pt = obs[-3:]
        dist_to_goal = np.linalg.norm(ref_pt - self._get_target_pos())
        reward =   -1 * self.REWARD_SCALE * dist_to_goal
        if dist_to_goal < self.EPSILON:
            done = True
            reward += self.SPARSE_REWARD
        else:
            done = False
        return obs, reward, done, {'success' : float(done)}

class PegInsertSawyer(PegInsert):
    FRAME_SKIP = 1
    NSUBSTEPS = 20
    SKILL_DIM = 3
    TASK_DIM = 9 # peg end position, peg top position, grip base pos, but could include more (grip base, peg_top, etc.)
    EPSILON = 0.02
    REWARD_SCALE = 0.1
    SPARSE_REWARD = 100
    JOINT_RANDOMIZATION_LOW = None
    JOINT_RANDOMIZATION_HIGH = None
    NUM_JOINTS = None

    def __init__(self, relative_grip_pos=True, rand_init=False, valid_area=False, z_skill=False, xy_skill=False, xyz_skill=False, 
                    mass_coef=1.0, friction_coef=1.0):
        self.relative_grip_pos = relative_grip_pos
        self.rand_init = rand_init
        self.valid_area = valid_area
        self.z_skill = z_skill
        self.xy_skill = xy_skill
        self.xyz_skill = xyz_skill
        self.peg_length = 0.09
        if self.z_skill:
            # Shift the z-pos of the peg top offset to the beginning of the skill space.
            self.SKILL_DIM += 1
            self.AGENT_DIM -= 1
        if self.xy_skill:
            self.SKILL_DIM += 2
            self.AGENT_DIM -= 2
        if self.xyz_skill:
            self.SKILL_DIM += 3
            self.AGENT_DIM -= 3

        # Add fake control bounds for init
        self.control_low = -100*np.ones(self.NUM_JOINTS)
        self.control_high = 100*np.ones(self.NUM_JOINTS)

        super(PegInsertSawyer, self).__init__()
        # Now we have the initialized model. Change the action space etc to be scaled correctly.
        bounds = self.sim.model.actuator_ctrlrange[:self.NUM_JOINTS]
        self.control_low = np.copy(bounds[:, 0])
        self.control_high = np.copy(bounds[:, 1])
        self.action_space = gym.spaces.Box(low=-1*np.ones(self.NUM_JOINTS), high=np.ones(self.NUM_JOINTS), dtype=np.float32)
        
        if self.z_skill: # We also modify the observation space to make it include the clip range for the z skill
            self.observation_space.low[self.AGENT_DIM] = -self.peg_length # first component of the skill space is now the z coordinate.
            self.observation_space.high[self.AGENT_DIM] = self.peg_length
        if self.xy_skill:
            self.observation_space.low[self.AGENT_DIM:self.AGENT_DIM+2] = -self.peg_length
            self.observation_space.high[self.AGENT_DIM:self.AGENT_DIM+2] = self.peg_length
        if self.xyz_skill:
            self.observation_space.low[self.AGENT_DIM:self.AGENT_DIM+3] = -self.peg_length
            self.observation_space.high[self.AGENT_DIM:self.AGENT_DIM+3] = self.peg_length

        # Change Coefficients in the sim
        if mass_coef != 1.0:
            self.sim.model.body_mass[:] *= mass_coef
        if friction_coef != 1.0:
            self.sim.model.geom_friction[:] *= friction_coef

    def _get_peg_obs(self, skill_obs):
        return skill_obs[-3:]

    def _get_target_pos(self):
        return self.sim.data.get_site_xpos('target')

    def display_skill(self, skill):
        if self.z_skill:
            self.model.body_pos[-1][:3] = skill[1:4]
        elif self.xy_skill:
            x, y = skill[0], skill[1]            
            if x**2 + y**2 < self.peg_length**2:
                perp_vec = np.cross(np.array([0,0,1]), np.array([x,y,0]))
                axis = perp_vec / np.linalg.norm(perp_vec)
                angle = np.pi/2 - np.arccos(np.sqrt(x**2+y**2) / self.peg_length)
                axis /= np.linalg.norm(axis)
                quat = np.concatenate([[np.cos(angle / 2.)], np.sin(angle / 2.) * axis])
                quat /= np.linalg.norm(quat)                
                self.model.body_quat[-1][:] = quat
            self.model.body_pos[-1][:3] = skill[2:5]
        elif self.xyz_skill:
            x, y = skill[0], skill[1]            
            if x**2 + y**2 < self.peg_length**2:
                perp_vec = np.cross(np.array([0,0,1]), np.array([x,y,0]))
                axis = perp_vec / np.linalg.norm(perp_vec)
                angle = np.pi/2 - np.arccos(np.sqrt(x**2+y**2) / self.peg_length)
                axis /= np.linalg.norm(axis)
                quat = np.concatenate([[np.cos(angle / 2.)], np.sin(angle / 2.) * axis])
                quat /= np.linalg.norm(quat)                
                self.model.body_quat[-1][:] = quat
            self.model.body_pos[-1][:3] = skill[3:6]
        else:
            self.model.body_pos[-1][:3] = skill[:3]
        
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
            grip_base = grip_base - ref_pt
        if self.xy_skill:
            peg_top = np.array([peg_top[2], peg_top[0], peg_top[1]])
        return np.concatenate([
                self.get_qpos(),
                self.get_qvel(),
                grip_base,
                peg_top,
                ref_pt
                ], axis=0)
    
    def in_valid_area(self):
        x_ref, y_ref, z_ref = self.get_site_com('ref_pt')
        if x_ref < 0.15 or abs(y_ref) > 0.5 or z_ref > 0.9 or z_ref < -0.25:
            return False
        else:
            return True

    def step(self, action):
        action = self.scale_action(action)
        obs, reward, done, info = super(PegInsertSawyer, self).step(action)
        if self.valid_area and not self.in_valid_area():
            done = True
        return obs, reward, done, info

    def reset(self):
        self.sim.reset()
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

class Insert_Sawyer5Arm1(PegInsertSawyer):
    JOINT_RANDOMIZATION_LOW = [-0.5, -1.25, -1.0, -1.0, -0.5]
    JOINT_RANDOMIZATION_HIGH = [0.5, 0.25, 1.75, 1.0, 0.5]
    ASSET = 'sawyer/5dof_robot1_peg.xml'
    AGENT_DIM = 5*2 + 6
    NUM_JOINTS = 5

class Insert_Sawyer5Arm2(PegInsertSawyer):
    JOINT_RANDOMIZATION_LOW = [-0.5, -1.25, -1.0, -1.0, -0.5]
    JOINT_RANDOMIZATION_HIGH = [0.5, 0.25, 1.75, 1.0, 0.5]
    ASSET = 'sawyer/5dof_robot2_peg.xml'
    AGENT_DIM = 5*2 + 6
    NUM_JOINTS = 5

class Insert_Sawyer5Arm1Rot(PegInsertSawyer):
    JOINT_RANDOMIZATION_LOW = [-1.0, -2.0, -1.5, -1.5, -1.0]
    JOINT_RANDOMIZATION_HIGH = [1.0, 0.5, 2.3, 1.5, 1.0]
    ASSET = 'sawyer/5dof_robot1_peg_rot.xml'
    AGENT_DIM = 5*2 + 6
    NUM_JOINTS = 5

class Insert_Sawyer6Arm1(PegInsertSawyer):
    ASSET = 'sawyer/6dof_robot1_peg.xml'
    JOINT_RANDOMIZATION_LOW = [-0.5, -1.25, -1.0, -1.0, -1.0, -0.5]
    JOINT_RANDOMIZATION_HIGH = [0.5, 0.25, 1.75, 1.0, 1.0, 0.5]
    AGENT_DIM = 6*2 + 6
    NUM_JOINTS = 6

class Insert_Sawyer6Arm2(PegInsertSawyer):
    ASSET = 'sawyer/6dof_robot2_peg.xml'
    JOINT_RANDOMIZATION_LOW = [-0.5, -1.25, -1.0, -1.0, -1.0, -0.5]
    JOINT_RANDOMIZATION_HIGH = [0.5, 0.25, 1.75, 1.0, 1.0, 0.5]
    AGENT_DIM = 6*2 + 6
    NUM_JOINTS = 6

class Insert_Sawyer7Arm1(PegInsertSawyer):
    ASSET = 'sawyer/7dof_robot1_peg.xml'
    JOINT_RANDOMIZATION_LOW = [-0.5, -1.25, -1.0, -1.0, -1.0, -1.0, -0.5]
    JOINT_RANDOMIZATION_HIGH = [0.5, 0.25, 1.75, 1.0, 1.0, 1.0, 0.5]
    AGENT_DIM = 7*2 + 6
    NUM_JOINTS = 7
