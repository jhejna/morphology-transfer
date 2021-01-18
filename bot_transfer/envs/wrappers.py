"""
Hierarchical Wrappers for the environments.
"""
import numpy as np
import gym
import os
from .base import Env, GoalEnv, convert_observation_to_space
import bot_transfer
from bot_transfer.utils.loader import load_from_name, LOW_LEVELS
import bot_transfer

class L2Low(gym.Wrapper):
    
    def __init__(self, env, delta_max=None, # delta max is a list with one arg per skill dim of max distance for sampling.
                            epsilon=0.25, # error bound for when sub goal is considered reached
                            sparse_reward=25, # Sparse reward for reaching subgoal
                            reward_scale=0.1, # scale of reward
                            survive_reward=0.0, # constant reward at each time step.
                            early_termination=False, # finish the episode when the goal is reached, or fixed length.
                            action_penalty=0.0, # penalty on action sizes.
                            reset_prob=1.0, # probability we actually reset the underlying env instead of sampling a new goal.
                            reset_free_limit=10, # max number of sub-goal episodes without env reset
                            relative=False, # whether or not to use relative state space information
                            use_her=False, # Whether or not to provide HER like observations.
                            additive_goals=True, # whether to sample goals using the range from delta_max or by resampling the global goal space.
                            goal_range_low=None, # Low range of valid skill-space goals
                            goal_range_high=None # High range of valid skill-space goals
                            ):
        assert issubclass(type(env), Env), "Error, provided environment does not inherit from base.Env"
        super(L2Low, self).__init__(env)
        
        # Save the goal sampling information
        self.delta_max = np.array(delta_max)
        self.additive_goals = additive_goals
        if not goal_range_high is None and not goal_range_low is None:
            assert len(goal_range_low) == self.env.SKILL_DIM and len(goal_range_high) == self.env.SKILL_DIM
            self.goal_range = gym.spaces.Box(low=np.array(goal_range_low), high=np.array(goal_range_high))
        else:
            self.goal_range = None
        
        # Save the reward information:
        self.epsilon = epsilon
        self.sparse_reward = sparse_reward
        self.reward_scale = reward_scale
        self.survive_reward = survive_reward
        self.action_penalty = action_penalty

        # Save the reset information
        self.early_termination = early_termination
        self.reset_prob = reset_prob
        self.reset_free_limit = reset_free_limit
        self.reset_free_count = 0 # initialize reset counter

        # Save the observation information
        self.use_her = use_her
        if self.use_her: # If we use HER, we can't use relative in the wrapper. Although, we should accept the parameter.
            self.relative = False
        else:
            self.relative = relative
        
        # Init state parameters
        self.last_env_obs = None # NOTE: May have to alter to reset here so we get last_env_obs if they call step before reset
        self.goal = None
        self.env_done = True # keep track of if the underlying env has ended for some reason.
        # Get the correct observation space by converting the environment's space.

        # Reset the environment and get the correct action
        self.reset()
        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done
        self.observation_space = convert_observation_to_space(observation)

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute L2 reward
        dist_to_goal = np.linalg.norm(desired_goal - achieved_goal)
        reward = -1 * self.reward_scale * dist_to_goal
        if dist_to_goal < self.epsilon:
            reward += self.sparse_reward
        # Add the action penalty and survive rewards
        reward -= self.action_penalty * np.sum(np.square(info['action']))
        reward += self.survive_reward
        return reward

    def _get_obs(self, env_obs):
        agent_obs = self.env.agent_obs(env_obs)
        if self.relative:
            skill_obs = self.goal - self.env.skill_obs(env_obs)
            goal_obs = np.zeros((self.env.SKILL_DIM,)) # the goal is always zero.
        else: # We provide state in absolute coordinates.
            skill_obs = self.env.skill_obs(env_obs)
            goal_obs = self.goal
        # support HER style low level
        if self.use_her:
            obs = {'observation': agent_obs,
                   'achieved_goal' : skill_obs,
                   'desired_goal' : goal_obs
                }
        else:
            obs = np.concatenate([agent_obs, skill_obs, goal_obs], axis=0)
        return obs

    def step(self, action):
        self.last_env_obs, _, self.env_done, info = self.env.step(action)
        # This is a gym goal env.
        info['action'] = action # Gym goal env says to store extra info needed for reward in info dict.
        skill_obs = self.env.skill_obs(self.last_env_obs)
        reward = self.compute_reward(skill_obs, self.goal, info)
        done = self.env_done
        if np.linalg.norm(self.goal - skill_obs) < self.epsilon:
            if self.early_termination:
                done = True
            info['success'] = 1.0
        else:
            info['success'] = 0.0
        obs = self._get_obs(self.last_env_obs)
        return obs, reward, done, info

    def reset(self, goal=None, **kwargs):
        # only call the environment reset with a certain probability or if we reach max reset free count.
        if self.env_done or self.last_env_obs is None or self.reset_free_count == self.reset_free_limit or self.np_random.rand() < self.reset_prob:
            # Under these conditions, we must reset the actual environment
            self.last_env_obs = self.env.reset(**kwargs) 
            self.reset_free_count = 0
        else:
            self.reset_free_count += 1 # otherwise increment the counter.
        if goal:
            self.goal = goal
        else:
            if self.additive_goals: # If additive goals, we sample a delta and add it to current position.
                self.goal = self.np_random.uniform(low=-self.delta_max, high=self.delta_max, size=self.env.SKILL_DIM) + self.skill_obs(self.last_env_obs)
                # If a goal range is specified, enforce it up to 500 attempts.
                if not self.goal_range is None:
                    num_attempts = 0
                    while not self.goal_range.contains(self.goal) and num_attempts < 500:
                        self.goal = self.np_random.uniform(low=-self.delta_max, high=self.delta_max, size=self.env.SKILL_DIM) + self.skill_obs(self.last_env_obs)
                        num_attempts += 1
                    # If we can never  get a good goal sample, we sample absolutely.
                    if not self.goal_range.contains(self.goal):
                        self.goal = self.goal_range.sample() 
            else:
                # If goals are not additive, we sample absolutely.
                self.goal = self.goal_range.sample() 
            # Clip the goal based on the original env action space.
            orig_sp = self.env.observation_space['observation'] if isinstance(self.env.observation_space, gym.spaces.Dict) else self.env.observation_space
            self.goal = np.clip(self.goal, orig_sp.low[self.env.AGENT_DIM:self.env.AGENT_DIM+self.env.SKILL_DIM], 
                                        orig_sp.high[self.env.AGENT_DIM:self.env.AGENT_DIM+self.env.SKILL_DIM])
            
            # Special Relabeling for the xyz_skill env
            if hasattr(self, "xyz_skill") and self.xyz_skill:
                if self.goal[0]**2 + self.goal[1]**2 > 0.09**2:
                    self.goal[2] = 0
                else:
                    self.goal[2] = np.sqrt(0.09**2 - self.goal[0]**2 - self.goal[1]**2)


        self.display_skill(self.goal)
        return self._get_obs(self.last_env_obs)

class CosLow(gym.Wrapper):
    
    def __init__(self, env, delta_max=None, 
                            epsilon=0.25, 
                            sparse_reward=25, 
                            reward_scale=1.0,
                            survive_reward=0.0,
                            penalize_overshoot=False,
                            early_termination=False,
                            action_penalty=0.0,
                            reset_prob=1.0,
                            reset_free_limit=10,
                            relative=False,
                            additive_goals=True,
                            goal_range_low=None,
                            goal_range_high=None
                            ):
        assert issubclass(type(env), Env), "Error, provided environment does not inherit from base.Env"
        super(CosLow, self).__init__(env)
        # Goal sampling params
        self.delta_max = np.array(delta_max)
        self.additive_goals = additive_goals
        self.relative = relative

        # Reward params
        self.epsilon = epsilon
        self.sparse_reward = sparse_reward
        self.reward_scale = reward_scale
        self.action_penalty = action_penalty
        self.survive_reward = survive_reward
        self.penalize_overshoot = penalize_overshoot

        # Reset params
        self.early_termination = early_termination
        self.reset_prob = reset_prob
        self.reset_free_limit = reset_free_limit
        self.reset_free_count = 0
        
        # Setup
        self.last_env_obs = None # NOTE: May have to alter to reset here so we get last_env_obs if they call step before reset
        self.env_done = True
        self.goal = None

        # Get the correct observation space
        self.reset()
        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done
        self.observation_space = convert_observation_to_space(observation)

    def _get_obs(self, env_obs):
        if self.relative:
            obs = np.concatenate([self.env.agent_obs(env_obs),
                                  self.goal - self.env.skill_obs(env_obs),
                                  np.zeros((self.env.SKILL_DIM,))], axis=0)
        else:
            obs = np.concatenate([self.env.agent_obs(env_obs),
                                  self.env.skill_obs(env_obs),
                                  self.goal], axis=0)
        return obs

    def step(self, action):
        # first, get the last
        last_skill_obs = self.env.skill_obs(self.last_env_obs)
        self.last_env_obs, _, self.env_done, info = self.env.step(action)
        skill_obs = self.env.skill_obs(self.last_env_obs)
        goal_vector = self.goal - last_skill_obs
        movement_vector =  skill_obs - last_skill_obs
        movement_projection = np.dot(movement_vector, goal_vector) / np.linalg.norm(goal_vector)
        if self.penalize_overshoot:
            if movement_projection > np.linalg.norm(goal_vector):
                overshoot = movement_projection - np.linalg.norm(goal_vector)
                reward = movement_projection - overshoot
            else:
                reward = movement_projection
        else:
            reward = min(movement_projection, np.linalg.norm(goal_vector))
        # scale the reward by time.
        reward = self.reward_scale * reward / self.dt
        reward += self.survive_reward
        reward -= self.action_penalty * np.sum(np.square(action))

        done = self.env_done
        if np.linalg.norm(self.goal - skill_obs) < self.epsilon:
            reward += self.sparse_reward
            if self.early_termination:
                done = True

        return self._get_obs(self.last_env_obs), reward, done, info

    def reset(self, goal=None, **kwargs):
        # only call the environment reset with a certain probability or if we reach max reset free count.
        if self.env_done or self.last_env_obs is None or self.reset_free_count == self.reset_free_limit or self.np_random.rand() < self.reset_prob:
            self.last_env_obs = self.env.reset(**kwargs)
            self.reset_free_count = 0
        else:
            self.reset_free_count += 1        
        # always set a new goal
        if goal:
            self.goal = goal
        else:
            if self.additive_goals:
                self.goal = self.np_random.uniform(low=-self.delta_max, high=self.delta_max, size=self.env.SKILL_DIM) + self.skill_obs(self.last_env_obs)
                if not self.goal_range is None:
                    num_attempts = 0
                    while not self.goal_range.contains(self.goal) and num_attempts < 500:
                        self.goal = self.np_random.uniform(low=-self.delta_max, high=self.delta_max, size=self.env.SKILL_DIM) + self.skill_obs(self.last_env_obs)
                        num_attempts += 1
                    # If we can never  get a good goal sample, we sample absolutely.
                    if not self.goal_range.contains(self.goal):
                        self.goal = self.goal_range.sample() 
            else:
                self.goal = self.np_random.uniform(low=self.goal_range_low, high=self.goal_range_high, size=self.env.SKILL_DIM)
            # Clip the goal based on the original env action space.
            orig_sp = self.env.observation_space['observation'] if isinstance(self.env.observation_space, gym.spaces.Dict) else self.env.observation_space
            self.goal = np.clip(self.goal, orig_sp.low[self.env.AGENT_DIM:self.env.AGENT_DIM+self.env.SKILL_DIM], 
                                        orig_sp.high[self.env.AGENT_DIM:self.env.AGENT_DIM+self.env.SKILL_DIM])

        return self._get_obs(self.last_env_obs)

class High(gym.Wrapper):

    def __init__(self, env, delta_max=None, 
                            skip=10,
                            epsilon=0.25,
                            low_level=None,
                            relative=False, # relative refers to the low level policy being relative.
                            use_her=False, # right now isnt used for anything but might cause inconsistent parser if removed
                            goal_range_low=None,
                            goal_range_high=None,
                            render_mode=None,
                            best=True): 
        assert issubclass(type(env), Env), "Error, provided environment does not inherit from base.Env"
        super(High, self).__init__(env)
        self.is_goal_env = issubclass(type(env), gym.GoalEnv)
        self.skip = skip
        self.relative = relative
        self.epsilon = epsilon
        
        if not low_level.startswith('/'):
            low_level = os.path.join(LOW_LEVELS, low_level)

        while 'params.json' not in os.listdir(low_level):
            contents = [os.path.join(low_level, d) for d in os.listdir(low_level) if os.path.isdir(os.path.join(low_level, d))]
            assert len(contents) == 1, "Traversing down directory with multiple paths."
            low_level = os.path.join(low_level, contents[0])

        model, _, params = load_from_name(low_level, best=best, load_env=False, ret_params=True)
        self.predict_fn = model.predict

        # Now we need to enforce consistency between loaded env and this env.
        if 'relative' in params['env_wrapper_args'] and self.relative != params['env_wrapper_args']['relative']:
            print("Warning: env wrapper mismatch. Setting relative to", params['env_wrapper_args']['relative'])
            self.relative = params['env_wrapper_args']['relative']
        if 'epsilon' in params['env_wrapper_args'] and self.epsilon != params['env_wrapper_args']['epsilon']:
            print("Warning: env wrapper mismatch. Setting epsilon to", params['env_wrapper_args']['epsilon'])
            self.epsilon = params['env_wrapper_args']['epsilon']
        
        if delta_max is None: #'delta_max' in params['env_wrapper_args'] and delta_max != params['env_wrapper_args']['delta_max']:
            print("Warning: Delta Max not specified. Setting delta_max to", params['env_wrapper_args']['delta_max'])
            delta_max = params['env_wrapper_args']['delta_max']

        if goal_range_low is None and 'goal_range_low' in params['env_wrapper_args']:
            goal_range_low = params['env_wrapper_args']['goal_range_low']
        if goal_range_high is None and 'goal_range_high' in params['env_wrapper_args']:
            goal_range_high = params['env_wrapper_args']['goal_range_high']

        if not goal_range_high is None and not goal_range_low is None:
            assert len(goal_range_low) == self.env.SKILL_DIM and len(goal_range_high) == self.env.SKILL_DIM
            self.goal_range = gym.spaces.Box(low=np.array(goal_range_low), high=np.array(goal_range_high))
        else:
            self.goal_range = None
        
        # Set the action space appropriatly
        if len(delta_max) == 1:
            self.action_space = gym.spaces.Box(low=-delta_max[0], high=delta_max[0], shape=(self.env.SKILL_DIM,))
        else:
            self.action_space = gym.spaces.Box(low=-np.array(delta_max), high=np.array(delta_max))
        self.last_env_obs = None
        self.render_mode = render_mode

        # Get the correct observation space
        self.reset()
        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done
        self.observation_space = convert_observation_to_space(observation)

    def compute_reward(self, achieved_goal, desired_goal, info):
        assert self.is_goal_env, "Compute reward should only be called for a goal env"
        reward = 0.0
        for achieved_goal in info['achieved_goal']:
            reward += self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward
        
    def _get_obs(self, env_obs):
        if self.is_goal_env:
            return {'observation': self.env.task_obs(env_obs),
                    'achieved_goal' : env_obs['achieved_goal'],
                    'desired_goal' : env_obs['desired_goal']}
        else:
            return self.env.task_obs(env_obs)

    def step(self, action):
        if isinstance(self.skip, tuple):
            num_low_steps = self.np_random.rand_int(low=self.skip[0], high=self.skip[1])
        else:
            num_low_steps = self.skip
        
        reward = 0.0
        done = False
        if self.is_goal_env:
            info = {'achieved_goal': []}
        else:
            info = {}
        info['frames'] = list()
        
        # set the low_goal obs
        low_goal = self.env.skill_obs(self.last_env_obs) + action
        # Clip the goal according to the environment space and goal space
        orig_sp = self.env.observation_space['observation'] if isinstance(self.env.observation_space, gym.spaces.Dict) else self.env.observation_space
        low_goal = np.clip(low_goal, orig_sp.low[self.env.AGENT_DIM:self.env.AGENT_DIM+self.env.SKILL_DIM], 
                                       orig_sp.high[self.env.AGENT_DIM:self.env.AGENT_DIM+self.env.SKILL_DIM])
        if self.goal_range:
            low_goal = np.clip(low_goal, self.goal_range.low, self.goal_range.high)

        if hasattr(self.env, "xyz_skill"):
            # this hack needs to exist for the sawyer environments.
            if low_goal[0]**2 + low_goal[1]**2 > 0.09**2:
                low_goal[2] = 0
            else:
                low_goal[2] = np.sqrt(0.09**2 - low_goal[0]**2 - low_goal[1]**2)
        
        self.display_skill(low_goal)
        for _ in range(num_low_steps):
            # Convert the observation so it can be used by LL.
            agent_obs = self.env.agent_obs(self.last_env_obs)
            skill_obs = self.env.skill_obs(self.last_env_obs)
            if self.relative:
                skill_obs = low_goal - skill_obs
                goal_obs = np.zeros((self.env.SKILL_DIM))
            else:
                goal_obs = low_goal
            low_obs = np.concatenate([agent_obs, skill_obs, goal_obs], axis=0)
            low_action = self.predict_fn(low_obs)[0]
            # print("Low Action", low_action)
            self.last_env_obs, partial_reward, done, env_info = self.env.step(low_action)
            reward += partial_reward
            if env_info:
                info.update(env_info) # Pull the most recent info from the env.
            # If HER, log the achieved goals so we can recompute the rewards if needed.
            if self.is_goal_env:
                info['achieved_goal'].append(self.last_env_obs['achieved_goal'].copy())

            ################## DEBUG RENDERING #################
            # self.env.render()
            # if not self.render_mode is None:
            #     self.env.render(self.render_mode)
            # info['frames'].append(self.env.render('rgb_array'))
            # CHANGING SOMETHING IN THE FILE
            ####################################################

            # Evaluate to see if we are within epsilon of the goal
            if done or np.linalg.norm(self.env.skill_obs(self.last_env_obs) - low_goal) < self.epsilon:
                break

        return self._get_obs(self.last_env_obs), reward, done, info
    
    def reset(self, **kwargs):
        self.last_env_obs = self.env.reset(**kwargs)
        return self._get_obs(self.last_env_obs)
