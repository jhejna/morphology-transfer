import os
import numpy as np
import tempfile
import gym
import xml.etree.ElementTree as ET
from .base import Env

class Waypoint(Env):
    WAYPOINT_SIZE = None
    ARENA_SPACE_LOW = None
    ARENA_SPACE_HIGH = None
    WAYPOINT_DIM = None # The current location in ref to waypoint is skill_obs[:WAYPOINT_DIM] = obs[AGENT_DIM:AGENT_DIM+SKILL_DIM][:WAYPOINT_DIM]
    REWARD_SCALE = None
    SPARSE_REWARD = None
    VISUALIZE = False

    def __init__(self, model_path=None):
        self.waypoint = self.ARENA_SPACE_HIGH # Init to far away from agent start.
        self.num_reached = 0
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "assets", self.ASSET)
        if self.VISUALIZE:
            tree = ET.parse(model_path)
            world_body = tree.find(".//worldbody")
            _, xml_path = tempfile.mkstemp(text=True, suffix='.xml')
            waypoint_elem = ET.Element('body')
            waypoint_elem.set("name", "waypoint")
            waypoint_elem.set("pos", "0 0 " + str(self.WAYPOINT_SIZE))
            waypoint_geom = ET.SubElement(waypoint_elem, "geom")
            waypoint_geom.set("conaffinity", "0")
            waypoint_geom.set("contype", "0")
            waypoint_geom.set("name", "waypoint")
            waypoint_geom.set("pos", "0 0 0")
            waypoint_geom.set("rgba", "0.2 0.9 0.2 0.8")
            waypoint_geom.set("size", str(self.WAYPOINT_SIZE))
            waypoint_geom.set("type", "sphere")
            world_body.insert(-1, waypoint_elem)
            tree.write(xml_path)
        else:
            xml_path = model_path
        
        super(Waypoint, self).__init__(model_path=xml_path)

    def _get_obs(self):
        return NotImplemented

    def _get_waypoint(self, skill_obs):
        return NotImplemented

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        dist_to_waypoint = np.linalg.norm(self.waypoint - self._get_waypoint(self.skill_obs(obs)))
        reward = -1*self.REWARD_SCALE * dist_to_waypoint
        if dist_to_waypoint < self.WAYPOINT_SIZE:
            reward += self.SPARSE_REWARD
            self.num_reached += 1
            self.waypoint = self.np_random.uniform(low=self.ARENA_SPACE_LOW, high=self.ARENA_SPACE_HIGH)
            if self.VISUALIZE:
                self.model.body_pos[-2][:self.WAYPOINT_DIM] = self.waypoint
            # update the observation to include the next waypoint
            obs = self._get_obs()
        return obs, reward, False, {'num_reached' : self.num_reached}
        
class WaypointNav(Waypoint):
    ARENA_SPACE_LOW = np.array([-8.0, -8.0])
    ARENA_SPACE_HIGH = np.array([8.0, 8.0])
    SKILL_DIM = 2
    WAYPOINT_DIM = 2
    TASK_DIM = 4 # agent position, waypoint position
    WAYPOINT_SIZE = 0.5
    REWARD_SCALE = 0.01
    SPARSE_REWARD = 100
    VISUALIZE = True

    def _get_waypoint(self, skill_obs):
        return skill_obs[:self.WAYPOINT_DIM]

class Waypoint_PointMass(WaypointNav):
    ASSET = 'point_mass.xml'
    AGENT_DIM = 2
    FRAME_SKIP = 3

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qvel.flat[:],
            self.get_body_com("torso")[:2],
            self.waypoint,
        ])
    
    def reset(self):
        self.sim.reset()
        self.waypoint = self.np_random.uniform(low=self.ARENA_SPACE_LOW, high=self.ARENA_SPACE_HIGH)
        if self.VISUALIZE:
            self.model.body_pos[-2][:self.WAYPOINT_DIM] = self.waypoint
        qpos = self.init_qpos + self.np_random.uniform(low=-self.WAYPOINT_SIZE, high=self.WAYPOINT_SIZE, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.2, high=0.2, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

class Waypoint_Ant(WaypointNav):
    ASSET = 'ant.xml'
    AGENT_DIM = 123
    FRAME_SKIP = 5
    
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:],
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.get_body_com("torso")[:2],
            self.waypoint,
        ])

    def step(self, action):
        obs, reward, done, info = super(Waypoint_Ant, self).step(action)
        if not np.isfinite(obs).all() or obs[0] > 1.0 or obs[0] < 0.2:
            done = True
        return obs, reward, done, info 

    def reset(self):
        self.sim.reset()
        self.waypoint = self.np_random.uniform(low=self.ARENA_SPACE_LOW, high=self.ARENA_SPACE_HIGH)
        if self.VISUALIZE:
            self.model.body_pos[-2][:self.WAYPOINT_DIM] = self.waypoint
        qpos = self.np_random.uniform(size=self.model.nq, low=-.1, high=.1) + self.init_qpos
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

class Waypoint_Quadruped(WaypointNav):
    ASSET = 'quadruped.xml'
    AGENT_DIM = 163
    FRAME_SKIP = 5
    
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:],
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.get_body_com("torso")[:2],
            self.waypoint,
        ])

    def step(self, action):
        obs, reward, done, info = super(Waypoint_Quadruped, self).step(action)
        # TODO: determine appropriate height for quadruped.
        # if not np.isfinite(obs).all() or obs[0] > 1.0 or obs[0] < 0.2:
        #     done = True
        return obs, reward, done, info 

    def reset(self):
        self.sim.reset()
        self.waypoint = self.np_random.uniform(low=self.ARENA_SPACE_LOW, high=self.ARENA_SPACE_HIGH)
        if self.VISUALIZE:
            self.model.body_pos[-2][:self.WAYPOINT_DIM] = self.waypoint
        qpos = self.np_random.uniform(size=self.model.nq, low=-.1, high=.1) + self.init_qpos
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

class WaypointSawyer(Waypoint):
    FRAME_SKIP = 1
    NSUBSTEPS = 20
    ARENA_SPACE_LOW = np.array([0.35, -0.28, -0.05])
    ARENA_SPACE_HIGH = np.array([0.7, 0.28, 0.25]) # USED TO BE 0.85
    SKILL_DIM = 3
    WAYPOINT_DIM = 3
    TASK_DIM = 9 # peg end position, peg top position, goal position, but could include more (grip base, peg_top, etc.)
    WAYPOINT_SIZE = 0.025
    REWARD_SCALE = 0.1
    SPARSE_REWARD = 100
    VISUALIZE = True
    JOINT_RANDOMIZATION_LOW = None
    JOINT_RANDOMIZATION_HIGH = None
    NUM_JOINTS = None

    def __init__(self, relative_grip_pos=True, rand_init=False, valid_area=False, z_skill=False, xy_skill=False, xyz_skill=False,
                mass_coef=1.0, friction_coef=1.0):
        
        self.relative_grip_pos = relative_grip_pos
        self.rand_init = rand_init
        self.valid_area = valid_area
        self.peg_length = 0.09
        self.z_skill = z_skill
        self.xy_skill = xy_skill
        self.xyz_skill = xyz_skill
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

        # Remove the table for the reach task.
        xml_path = os.path.join(os.path.dirname(__file__), "assets", self.ASSET)
        tree = ET.parse(xml_path)

        # Test for Mesh Directory.
        compiler_elem = tree.find(".//compiler")
        if compiler_elem.get('meshdir'):
            compiler_elem.set('meshdir', os.path.join(os.path.dirname(xml_path), compiler_elem.get('meshdir')))

        world_body_elem = tree.find(".//worldbody")            
        for body in world_body_elem.findall('body'):
            if body.get('name') == 'table':
                world_body_elem.remove(body)
        _, xml_path = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(xml_path)

        # Add fake control bounds for init
        self.control_low = -100*np.ones(self.NUM_JOINTS)
        self.control_high = 100*np.ones(self.NUM_JOINTS)

        super(WaypointSawyer, self).__init__(model_path=xml_path)
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

    def _get_waypoint(self, skill_obs):
        return skill_obs[-self.WAYPOINT_DIM:]

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
        peg_top = self.get_site_com('peg_top') - ref_pt #NOTE: Length of the peg is 0.09 (between the two sites)
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
            ref_pt,
            self.waypoint
        ])
    
    def in_valid_area(self):
        x_ref, y_ref, z_ref = self.get_site_com('ref_pt')
        if x_ref < 0.15 or abs(y_ref) > 0.5 or z_ref > 0.9 or z_ref < -0.25:
            return False
        else:
            return True

    def step(self, action):
        action = self.scale_action(action)
        obs, reward, done, info = super(WaypointSawyer, self).step(action)
        if self.valid_area and not self.in_valid_area():
            done = True
        return obs, reward, done, info

    def reset(self):
        self.sim.reset()
        self.waypoint = self.np_random.uniform(low=self.ARENA_SPACE_LOW, high=self.ARENA_SPACE_HIGH)
        if self.VISUALIZE:
            self.model.body_pos[-2][:self.WAYPOINT_DIM] = self.waypoint
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

class Waypoint_Sawyer5Arm1(WaypointSawyer):
    JOINT_RANDOMIZATION_LOW = [-0.5, -1.25, -1.0, -1.0, -0.5]
    JOINT_RANDOMIZATION_HIGH = [0.5, 0.25, 1.75, 1.0, 0.5]
    ASSET = 'sawyer/5dof_robot1_peg.xml'
    AGENT_DIM = 5*2 + 6
    NUM_JOINTS = 5

class Waypoint_Sawyer5Arm1Rot(WaypointSawyer):
    JOINT_RANDOMIZATION_LOW = [-1.0, -2.0, -1.5, -1.5, -1.0]
    JOINT_RANDOMIZATION_HIGH = [1.0, 0.5, 2.3, 1.5, 1.0]
    ASSET = 'sawyer/5dof_robot1_peg_rot.xml'
    AGENT_DIM = 5*2 + 6
    NUM_JOINTS = 5

class Waypoint_Sawyer5Arm2(WaypointSawyer):
    JOINT_RANDOMIZATION_LOW = [-0.5, -1.25, -1.0, -1.0, -0.5]
    JOINT_RANDOMIZATION_HIGH = [0.5, 0.25, 1.75, 1.0, 0.5]
    ASSET = 'sawyer/5dof_robot2_peg.xml'
    AGENT_DIM = 5*2 + 6
    NUM_JOINTS = 5

class Waypoint_Sawyer6Arm1(WaypointSawyer):
    ASSET = 'sawyer/6dof_robot1_peg.xml'
    JOINT_RANDOMIZATION_LOW = [-0.5, -1.25, -1.0, -1.0, -1.0, -0.5]
    JOINT_RANDOMIZATION_HIGH = [0.5, 0.25, 1.75, 1.0, 1.0, 0.5]
    AGENT_DIM = 6*2 + 6
    NUM_JOINTS = 6

class Waypoint_Sawyer6Arm2(WaypointSawyer):
    ASSET = 'sawyer/6dof_robot2_peg.xml'
    JOINT_RANDOMIZATION_LOW = [-0.5, -1.25, -1.0, -1.0, -1.0, -0.5]
    JOINT_RANDOMIZATION_HIGH = [0.5, 0.25, 1.75, 1.0, 1.0, 0.5]
    AGENT_DIM = 6*2 + 6
    NUM_JOINTS = 6

class Waypoint_Sawyer7Arm1(WaypointSawyer):
    ASSET = 'sawyer/7dof_robot1_peg.xml'
    JOINT_RANDOMIZATION_LOW = [-0.5, -1.25, -1.0, -1.0, -1.0, -1.0, -0.5]
    JOINT_RANDOMIZATION_HIGH = [0.5, 0.25, 1.75, 1.0, 1.0, 1.0, 0.5]
    AGENT_DIM = 7*2 + 6
    NUM_JOINTS = 7
