import os
import imageio
import copy
import gym
from stable_baselines.bench import Monitor
from stable_baselines import logger
import numpy as np
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.misc_util import mpi_rank_or_zero
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.callbacks import BaseCallback
import bot_transfer
from bot_transfer.utils.loader import get_paths, get_env, get_alg, get_policy, setup_env
from bot_transfer.utils.loader import load, HIGH_LEVELS, LOW_LEVELS, ModelParams
from bot_transfer.utils.tester import eval_policy

class TrainCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, params, data_dir, tb_dir=None, verbose=1, eval_env=True):
        super(TrainCallback, self).__init__(verbose)
        self.params = params
        self.checkpoint_freq = params['checkpoint_freq']
        self.eval_env = eval_env
        self.eval_freq = params['eval_freq']
        self.data_dir = data_dir
        self.tb_dir = tb_dir
        self.best_mean_reward = -np.inf
        self.save_path = os.path.join(data_dir, 'best_model')

    def _on_step(self) -> bool:
        # NOTE: can add custom tensorboard callbacks here if wanted to log extra materials

        if self.n_calls % self.eval_freq == 0:
            if self.eval_env:
                # procedure for eval when we load the env.
                env = get_env(self.params)
                env = setup_env(env, self.params)
                mean_reward, _ = eval_policy(self.model, env, num_ep=100, deterministic=True, verbose=0, gif=False)
                env.close()
                del env # make sure we free the memory
            else:
                # Retrieve training reward
                x, y = ts2xy(load_results(self.data_dir), 'timesteps')
                if len(x) > 0:
                    # Mean training reward over the last 100 episodes
                    mean_reward = np.mean(y[-100:])
                else:
                    mean_reward = -np.inf
            if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
            
            # New best model, you could save the agent here
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Example for saving best model
                if self.verbose > 0:
                    print("Saving new best model.")
                self.model.save(self.data_dir + '/best_model')

        if self.checkpoint_freq and self.n_calls % self.checkpoint_freq == 0:
            if self.verbose > 0:
                print("Saving Checkpoint for timestep", self.num_timesteps)
            self.model.save(self.data_dir + 'checkpoint_' + str(self.num_timesteps))

        return True

def run_train(params, model=None, env=None, path=None): 
    print("Training Parameters: ", params)

    data_dir, tb_path = get_paths(params, path=path)
    os.makedirs(data_dir, exist_ok=True)
    # Save parameters immediately
    params.save(data_dir)

    rank = mpi_rank_or_zero()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    # Create the environment if not given
    if env is None:  
        def make_env(i):
            env = get_env(params)
            info_keywords = tuple() # tuple(['success',])
            env = Monitor(env, data_dir + '/' + str(i), allow_early_resets=params['early_reset'], info_keywords=info_keywords)
            return env

        # env = DummyVecEnv([(lambda n: lambda: make_env(n))(i) for i in range(params['num_proc'])])
        env = make_env(0)
        if params['normalize']:
            env = VecNormalize(env)
    # Set the seeds
    if params['seed']:
        seed = params['seed'] + 100000 * rank
        set_global_seeds(seed)
        params['alg_args']['seed'] = seed

    if 'noise' in params and params['noise']:
        from stable_baselines.ddpg import OrnsteinUhlenbeckActionNoise
        n_actions = env.action_space.shape[-1]
        params['alg_args']['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(params['noise'])*np.ones(n_actions))
    
    if model is None:
        alg = get_alg(params)
        policy = get_policy(params)
        if params['use_her']:
            from bot_transfer.algs import HER
            model = HER(policy, env, alg, **params['her_args'], verbose=1, 
                        tensorboard_log=tb_path, policy_kwargs=params['policy_args'], **params['alg_args'])
        else:
            model = alg(policy,  env, verbose=1, tensorboard_log=tb_path, policy_kwargs=params['policy_args'], **params['alg_args'])
    else:
        model.set_env(env)

    print("\n===============================\n")
    print("TENSORBOARD PATH:", tb_path)
    print("\n===============================\n")

    callback = TrainCallback(params, data_dir, tb_path)

    model.learn(total_timesteps=params['timesteps'], log_interval=params['log_interval'], 
                callback=callback)
    
    model.save(data_dir +'/final_model')

    if params['normalize']:
        env.save(data_dir + '/environment.pkl')        
    env.close()

def train(params, path=None):
    low_level = params['low_level'] if 'low_level' in params else None
    high_level = params['high_level'] if 'high_level' in params else None
    
    alg = params['alg']

    discriminator_algs = ["DSAC"]
    kl_algs = ["KLSAC"]

    # Check for auto-setting environment names.
    # This works for:
    #   a) training high level policies on different tasks across agents
    #   b) not specifying the agent when finetuning high level policies.
    if not low_level is None and len(params['env'].split('_')) == 1:
        # Get the correct environment name from the low level name
        # Low level names come in the form: Waypoint_Sawyer6Arm1_L2Low_SAC_s4_0
        agent_str = os.path.basename(low_level).split('_')[1]
        params['env'] = '_'.join([params['env'], agent_str])
        print("NOTICE: Auto modified Env to", params['env'])

    # Check for auto-setting seed
    if params['seed'] is None and (not low_level is None or not high_level is None):
        # NOTE: Default to selecting the seed from high level for proper finetuning support.
        policy_with_seed = high_level if not high_level is None else low_level
        policy_name_components = os.path.basename(policy_with_seed).split('_')
        policy_seed_component = policy_name_components[-2]
        if policy_seed_component.startswith('s'):
            params['seed'] = int(policy_seed_component[1:])
            print("USING SEED", params['seed'])

    # Case 1: Training Low Level
    # High Level is None, Low Level is None
    # Execute regular training loop.
    if low_level is None and high_level is None:
        print("################################")
        print("## Launching Regular Training ##")
        print("################################")
        run_train(params, model=None, env=None, path=path)
    
    # Case 2: Training High Level.
    # Low Level is Policy, high level is None, alg is not discrim
    elif not low_level is None and high_level is None and not alg in discriminator_algs:
        print("###################################")
        print("## Launching High Level Training ##")
        print("###################################")
        params["env_wrapper_args"]["low_level"] = low_level
        del params["low_level"]
        run_train(params, model=None, env=None, path=path) 

    # Case 3: Training Discriminator
    # Low Level is Policy, Alg is DSAC
    elif not low_level is None and high_level is None and alg in discriminator_algs:
        print("######################################")
        print("## Launching Discriminator Training ##")
        print("######################################")
        params["alg_args"]["discrim_model"] = low_level
        del params["low_level"]
        run_train(params, model=None, env=None, path=path) 

    # Case 4: Finetuning High Level (REGULAR)
    # High level is Policy, Alg does not start with KL
    elif not low_level is None and not high_level is None and not alg in kl_algs:
        print("#####################################")
        print("## Launching High Level Finetuning ##")
        print("#####################################")
        # Load the model
        high_level_path = high_level
        if not high_level_path.startswith('/'):
                high_level_path = os.path.join(HIGH_LEVELS, high_level_path)
        while 'params.json' not in os.listdir(high_level_path):
            contents = [os.path.join(high_level_path, d) for d in os.listdir(high_level_path) if os.path.isdir(os.path.join(high_level_path, d))]
            assert len(contents) == 1, "Traversing down directory with multiple paths."
            high_level_path = os.path.join(high_level_path, contents[0])
        
        params["env_wrapper_args"]["low_level"] = low_level
        del params["low_level"]

        model, _ = load(high_level_path, params, best=True, load_env=False)
        run_train(params, model=model, env=None, path=path)

    # Case 5: Finetune KL High
    # High Level is Policy, Alg starts with KL
    elif not low_level is None and not high_level is None and alg in kl_algs:
        print("########################################")
        print("## Launching High Level KL-Finetuning ##")
        print("########################################")
        high_level_path = high_level
        if not high_level_path.startswith('/'):
                high_level_path = os.path.join(HIGH_LEVELS, high_level_path)
        while 'params.json' not in os.listdir(high_level_path):
            contents = [os.path.join(high_level_path, d) for d in os.listdir(high_level_path) if os.path.isdir(os.path.join(high_level_path, d))]
            assert len(contents) == 1, "Traversing down directory with multiple paths."
            high_level_path = os.path.join(high_level_path, contents[0])
        params["alg_args"]["kl_model"] = params["high_level"]
        del params["high_level"]
        params["env_wrapper_args"]["low_level"] = low_level
        del params["low_level"]
        model, _ = load(high_level_path, params, best=True, load_env=False)
        run_train(params, model=model, env=None, path=path)

    else:
        raise ValueError("Parameter specifications did not fit into one algorithm type.")
