import argparse
from bot_transfer.utils.loader import ModelParams

def boolean(item):
    if item == 'true' or item == 'True':
        return True
    elif item == 'false' or item == 'False':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

BASE_ARGS = {
    # Exclude Env and Alg as these are required for creating the params object.
    'timesteps' : int,
    'policy' : str,
    'early_reset' : boolean,
    'normalize' : boolean,
    'time_limit' : int,
    'seed' : int,
    'log_interval' : int,
    'tensorboard' : str,
    'name' : str,
    'num_proc' : str,
    'eval_freq' : int,
    'checkpoint_freq' : int,
    'use_her' : boolean,
    'high_level' : str,
    "low_level" : str,
}

ENV_ARGS = {
    'rand_init': boolean,
    'relative_grip_pos': boolean,
    'valid_area': boolean,
    'z_skill' : boolean,
    "xy_skill" : boolean,
    "xyz_skill": boolean,
    'mass_coef' : float,
    'friction_coef' : float,
}

ENV_WRAPPER_ARGS = {
    'delta_max' : (float, '+'),
    'epsilon' : float,
    'sparse_reward': float,
    'reward_scale' : float,
    'early_termination' : boolean,
    'action_penalty' : float,
    'reset_prob' : float,
    'reset_free_limit': int,
    'relative': boolean,
    'skip' : int,
    'survive_reward' : float,
    'goal_range_low' : (float, '+'),
    'goal_range_high' : (float, '+'),
    'additive_goals' : boolean,
    'best' : boolean,
}

ALG_ARGS = {
    "learning_rate": float,
    "batch_size": int,
    "buffer_size": int,
    "learning_starts": int,
    # Discrim Configuration
    "discrim_model": str,
    "discrim_buffer_size": int,
    "discrim_layers" : (int, '+'),
    "discrim_learning_rate": float,
    "discrim_start" : int,
    "discrim_weight" : float,
    "discrim_online" :boolean,
    "discrim_clip" : float,
    "discrim_batch_size": int,
    "discrim_train_freq": int,
    "discrim_stop": float,
    "discrim_coef": float,
    "discrim_decay": boolean,
    "discrim_time_limit": int,
    "discrim_include_skill": boolean,
    "discrim_include_next_state": boolean,
    "discrim_relative": boolean,

    # KL Configuration
    "kl_model" : str,
    "kl_coef" : float,
    "kl_type" : str,
    "kl_stop" : float,
    "kl_decay" : boolean,
}

HER_ARGS = {
    'n_sampled_goal': int,
    'goal_selection_strategy': int,
    'her_clip_goals' : boolean
}

POLICY_ARGS = {
    'layers' : (int, '+'),
}

def add_args_from_dict(parser, arg_dict):
    for arg_name, arg_type in arg_dict.items():
        arg_name = "--" + arg_name.replace('_', '-')
        if isinstance(arg_type, tuple) and len(arg_type) == 2:
            parser.add_argument(arg_name, type=arg_type[0], nargs=arg_type[1], default=None)
        else:
            parser.add_argument(arg_name, type=arg_type, default=None)

def base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    parser.add_argument("--env-wrapper", type=str)
    parser.add_argument("--alg", type=str)
    add_args_from_dict(parser, BASE_ARGS)
    return parser

def train_parser():
    parser = base_parser()
    add_args_from_dict(parser, ENV_ARGS)
    add_args_from_dict(parser, ENV_WRAPPER_ARGS)
    add_args_from_dict(parser, ALG_ARGS)
    add_args_from_dict(parser, HER_ARGS)
    add_args_from_dict(parser, POLICY_ARGS)
    return parser

def args_to_params(args):
    params = ModelParams(args.env, args.alg)
    for arg_name, arg_value in vars(args).items():
        if not arg_value is None:
            if arg_name in BASE_ARGS or arg_name in ("env", "alg", "env_wrapper"):
                params[arg_name] = arg_value
            elif arg_name in ENV_ARGS:
                params['env_args'][arg_name] = arg_value
            elif arg_name in ENV_WRAPPER_ARGS:
                params['env_wrapper_args'][arg_name] = arg_value
            elif arg_name in ALG_ARGS:
                params['alg_args'][arg_name] = arg_value
            elif arg_name in HER_ARGS:
                params['her_args'][arg_name] = arg_value
            elif arg_name in POLICY_ARGS:
                params['policy_args'][arg_name] = arg_value
            else:
                raise ValueError("Provided argument does not fit into categories")
    # Enforce uniformity over the HER arguments
    if params['use_her']:
        if params['env_wrapper']:
            params['env_wrapper_args']['use_her'] = params['use_her']
        if 'relative' in params['env_wrapper_args'] and params['env_wrapper_args']['relative']:
            params['her_args']['relative'] = params['env_wrapper_args']['relative']
        if 'goal_range_low' in params['env_wrapper_args']:
            params['her_args']['goal_range_low'] = params['env_wrapper_args']['goal_range_low']
        if 'goal_range_high' in params['env_wrapper_args']:
            params['her_args']['goal_range_high'] = params['env_wrapper_args']['goal_range_high']
    return params

def convert_kwargs_args(kwargs, parser):
    arg_list = []
    for key in kwargs.keys():
        arg_list.append('--' + key.replace('_', '-'))
        if isinstance(kwargs[key], list):
            arg_list.extend([str(item) for item in kwargs[key]])
        else:
            arg_list.append(str(kwargs[key]))
    args, unknown_args = parser.parse_known_args(arg_list)
    if len(unknown_args) > 0:
        print("############# ERROR #####################################")
        print("Unknown Arguments:", unknown_args)
        print("#########################################################")
    return args
