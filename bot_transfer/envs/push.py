import os
import numpy as np
import xml.etree.ElementTree as ET
import tempfile
from .base import Env


class Push(Env):

    PUSH_TARGET = np.array([-0.10, -0.08])
    SPARSE_REWARD = 100
    REWARD_SCALE = 0.5
    EPSILON = 0.025
    
    FRAME_SKIP = 3
    SKILL_DIM = 2
    TASK_DIM = 6 # Agent end effector, box position, box velocity
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "assets", self.ASSET)

        # Add in the target box location
        tree = ET.parse(model_path)
        world_body = tree.find(".//worldbody")
        _, xml_path = tempfile.mkstemp(text=True, suffix='.xml')
        push_dest_elem = ET.Element('body')
        push_dest_elem.set("name", "push_dest")
        push_dest_elem.set("pos", "%f %f 0.02" % (self.PUSH_TARGET[0], self.PUSH_TARGET[1]))
        push_dest_geom = ET.SubElement(push_dest_elem, "geom")
        push_dest_geom.set("conaffinity", "0")
        push_dest_geom.set("contype", "0")
        push_dest_geom.set("name", "push_dest")
        push_dest_geom.set("pos", "0 0 0")
        push_dest_geom.set("rgba", "0.2 0.2 0.9 0.3")
        push_dest_geom.set("size", "0.02 0.02 0.02")
        push_dest_geom.set("type", "box")
        world_body.insert(-2, push_dest_elem)
        tree.write(xml_path)
        super(Push, self).__init__(model_path=xml_path)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()

        dist_to_target = np.linalg.norm(self.get_body_com("target")[:2] - self.PUSH_TARGET)
        if dist_to_target < self.EPSILON:
            reward = self.SPARSE_REWARD
            done = True
        else:
            reward = -1*self.REWARD_SCALE*dist_to_target
            done = False

        return obs, reward, done, {'is_success': done}

class Push_PointMass(Push):
    RAND_INIT = False
    ASSET = 'reacher/push_pm.xml'
    AGENT_DIM = 2

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qvel.flat[:-2],
            self.get_body_com("fingertip")[:2],
            self.get_body_com("target")[:2],
            self.sim.data.qvel.flat[-2:],
        ])

    def reset(self):
        self.sim.reset()
        if self.RAND_INIT:
            qpos = self.np_random.uniform(low=-0.05, high=0.05, size=self.model.nq)
            qpos[0] -= 0.2 
            qpos += self.init_qpos
            qpos[-2:] = np.array([0.26, 0.26]) # Move the boxes out of the way
        else:
            qpos = self.np_random.uniform(low=-0.05, high=0.05, size=self.model.nq) + self.init_qpos
            qpos[-2:] = self.init_qpos[-2:]
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

class Empty_PointMass(Push_PointMass):
    RAND_INIT = True

class PushArm(Push):
    RAND_INIT = False
    RAND_LIMITS = None

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:-2] # remove position of target
        print(theta.shape)
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qvel.flat[:-2],
            self.get_body_com("fingertip")[:2],
            self.get_body_com("target")[:2],
            self.sim.data.qvel.flat[-2:],
        ])

    def reset(self):
        self.sim.reset()
        if self.RAND_INIT:
            rot_rand = self.np_random.uniform(low=-np.pi, high=np.pi, size=1)
            arm_rand = self.np_random.uniform(low=-self.RAND_LIMITS, high=self.RAND_LIMITS, size=self.model.nq-1)
            qpos = np.concatenate((rot_rand, arm_rand)) + self.init_qpos
            qpos[-2:] = np.array([0.26, 0.26])
        else:
            qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
            qpos[-2:] = self.init_qpos[-2:]
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        print(self._get_obs().shape)
        return self._get_obs()

class Push_2Link(PushArm):
    RAND_LIMITS = 3
    AGENT_DIM = 2*3
    ASSET = 'reacher/push_2link.xml'

class Empty_2Link(PushArm):
    RAND_LIMITS = 3
    RAND_INIT = True
    AGENT_DIM = 2*3
    ASSET = 'reacher/push_2link.xml'

class Push_3Link(PushArm):
    RAND_LIMITS = 2.7
    AGENT_DIM = 3*3
    ASSET = 'reacher/push_3link.xml'

class Empty_3Link(PushArm):
    RAND_LIMITS = 2.7
    AGENT_DIM = 3*3
    RAND_INIT = True
    ASSET = 'reacher/push_3link.xml'

class Push_4Link(PushArm):
    RAND_LIMITS = 2.0
    AGENT_DIM = 4*3
    ASSET = 'reacher/push_4link.xml'

class Empty_4Link(PushArm):
    RAND_LIMITS = 2.0
    AGENT_DIM = 4*3
    RAND_INIT = True
    ASSET = 'reacher/push_4link.xml'