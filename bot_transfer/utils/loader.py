import os
import bot_transfer
import json
from datetime import date
import stable_baselines

BASE = os.path.dirname(os.path.dirname(bot_transfer.__file__)) + '/data'
LOGS = os.path.dirname(os.path.dirname(bot_transfer.__file__)) + '/tb_logs'
HIGH_LEVELS = os.path.dirname(os.path.dirname(bot_transfer.__file__)) + '/high_levels'
LOW_LEVELS = os.path.dirname(os.path.dirname(bot_transfer.__file__)) + '/low_levels'

class ModelParams(dict):

    def __init__(self, env : str, alg : str, wrapper=None):
        super(ModelParams, self).__init__()
        # Construction Specification
        self['alg'] = alg
        self['env'] = env
        self['env_wrapper'] = wrapper
        self['policy'] = 'MlpPolicy' # TODO: Support different types of policies. Currently does not.
        self['use_her'] = False
        # Arg Dicts
        self['env_args'] = dict()
        self['env_wrapper_args'] = dict()
        self['alg_args'] = dict()
        self['her_args'] = dict()
        self['policy_args'] = dict()
        # Env Wrapper Arguments
        self['early_reset'] = True
        self['normalize'] = False
        self['time_limit'] = None
        # Training Args
        self['seed'] = None
        self['timesteps'] = 250000
        # Logistical Args
        self['log_interval'] = 20
        self['name'] = None
        self['tensorboard'] = None
        self['num_proc'] = 1 # Default to single process
        self['eval_freq'] = 100000
        self['checkpoint_freq'] = None
    
    def get_save_name(self) -> str:
        if self['name']:
            name =  self['name']
        else:
            print("ENV WRAP", self['env_wrapper'])
            name = self['env'] + ('_' + self['env_wrapper'] if self['env_wrapper'] else "") + '_' + self['alg']
        if not self['seed'] is None:
            name += '_s' + str(self['seed'])
        return name

    def save(self, path : str):
        if path.endswith(".json"):
            with open(path, 'w') as fp:
                json.dump(self, fp, indent=4)
        else:
            with open(os.path.join(path, 'params.json'), 'w') as fp:
                json.dump(self, fp, indent=4)

    @classmethod
    def load(cls, path):
        if not path.startswith('/'):
            path = os.path.join(BASE, path)
        if os.path.isdir(path) and 'params.json' in os.listdir(path):
            path = os.path.join(path, 'params.json')
        elif os.path.exists(path):
            pass
        else:
            raise ValueError("Params file not found in specified save directory:" + str(path))
        with open(path, 'r') as fp:
            data = json.load(fp)
        params = cls(data['env'], data['alg'])
        params.update(data)
        return params

def get_alg(params: ModelParams):
    alg_name = params['alg']
    try:
        alg = vars(bot_transfer.algs)[alg_name]
    except:
        alg = vars(stable_baselines)[alg_name]
    return alg

def get_env(params: ModelParams):
    env_name = params['env']
    try:
        env_cls = vars(bot_transfer.envs)[params['env']]
        env = env_cls(**params['env_args'])
        if params['env_wrapper']:
            env = vars(bot_transfer.envs)[params['env_wrapper']](env, **params['env_wrapper_args'])
        if params['time_limit']:
            from gym.wrappers import TimeLimit
            env = TimeLimit(env, params['time_limit'])
    except:
        # If we don't get the env, then we assume its a gym environment
        import gym
        env = gym.make(params['env'])
        if params['env_wrapper']:
            env = vars(gym.wrappers)[params['env_wrapper']](env, **params['env_wrapper_args'])
    return env

def setup_env(env, params):
    # Takes care of configuration for inference. This exists in order to provide a clean interface to loading HER policies.
    # Fixes so far related to HER.
    if params['use_her'] and 'relative' in params['env_wrapper_args'] and params['env_wrapper_args']['relative']:
        # Relative was only applied in training loop, so we need to convert to relative if it was trained with relative.
        using_l2_low = False
        wrapper_depth = 0
        try:
            if isinstance(env, bot_transfer.envs.L2Low):
                using_l2_low = True
                wrapper_depth = 1
            elif isinstance(env.env, bot_transfer.envs.L2Low):
                using_l2_low = True
                wrapper_depth = 2
            elif isinstance(env.env.env, bot_transfer.envs.L2Low):
                using_l2_low = True
                wrapper_depth = 3
        except:
            pass
        if using_l2_low:
            print("#######################################")
            print("# Notice: Using relative observations #")
            print("#######################################")
            if wrapper_depth == 1:
                env.relative = True
            if wrapper_depth == 2:
                env.env.relative = True
            if wrapper_depth == 3:
                env.env.env.relative = True

            env.relative = True
        # we need to wrap the environment with the HER wrapper.
    if params['use_her']:
        env = bot_transfer.algs.HERGoalEnvWrapper(env)
    return env

def get_policy(params: ModelParams):
    policy_name = params['policy']
    try:
        policy = vars(bot_transfer.policies)[policy_name]
        return policy
    except:
        alg_name = params['alg']
        if 'SAC' in alg_name:
            search_location = stable_baselines.sac.policies
        elif 'DDPG' in alg_name:
            search_location = stable_baselines.ddpg.policies
        elif'DQN' in alg_name:
            search_location = stable_baselines.deepq.policies
        elif 'TD3' in alg_name:
            search_location = stable_baselines.td3.policies
        else:
            search_location = stable_baselines.common.policies
        policy = vars(search_location)[policy_name]
        return policy
    
def get_paths(params: ModelParams, path=None):
    date_prefix = date.today().strftime('%m_%d_%y')
    if path:
        date_dir = os.path.join(path, date_prefix)
    else:
        date_dir = os.path.join(BASE, date_prefix)
    save_name = params.get_save_name()
    if os.path.isdir(date_dir):
        candidates = [f_name for f_name in os.listdir(date_dir) if '_'.join(f_name.split('_')[:-1]) == save_name]
        if len(candidates) == 0:
            save_name += '_0'
        else:
            num = max([int(dirname[-1]) for dirname in candidates]) + 1
            save_name += '_' + str(num)
    else:
        save_name += '_0'
    
    save_path = os.path.join(date_dir, save_name)
    tb_path = os.path.join(LOGS, date_prefix, save_name) if params['tensorboard'] else None
    return save_path, tb_path

def load_from_name(path, best=False, load_env=True, ret_params=False):
    if not path.startswith('/'):
        path = os.path.join(BASE, path)
    params = ModelParams.load(path)
    if ret_params:
        return load(path, params, best=best, load_env=load_env) + (params,)
    return load(path, params, best=best, load_env=load_env)

def load(path: str, params : ModelParams, best=False, load_env=True):
    if not path.startswith('/'):
        path = os.path.join(BASE, path)
    files = os.listdir(path)
    if not 'final_model.zip' in files and 'best_model.zip' in files:
        model_path = path + '/best_model.zip'
    elif 'best_model.zip' in files and best:
        model_path = path + '/best_model.zip'
    elif 'final_model.zip' in files:
        model_path = path + '/final_model.zip'
    else:
        raise ValueError("Cannot find a model for name: " + path)
    # get model
    alg = get_alg(params) if not params['use_her'] else bot_transfer.algs.HER
    model = alg.load(model_path, **params['alg_args'])
    if load_env:
        env = get_env(params)
        env = setup_env(env, params)
    else:
        env = None
    return model, env
