import os
import copy
import numpy as np
import bot_transfer
from .loader import load_from_name, load
from .loader import BASE, LOW_LEVELS, HIGH_LEVELS, ModelParams

RENDERS = os.path.dirname(os.path.dirname(bot_transfer.__file__)) + '/renders'

def eval_policy(model, env, num_ep=10, deterministic=True, verbose=1, gif=False, render=False):
    ep_rewards, ep_lens, ep_infos = list(), list(), list()
    mode = 'rgb_array' if gif else 'human'
    frames = list()
    for ep_index in range(num_ep):
        obs = env.reset()
        done = False
        ep_rew, ep_len = 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            ep_len += 1
            ep_rew += reward
            if 'frames' in info:
                frames.extend(info['frames'])
            if render:
                frames.append(env.render(mode=mode))
            
        ep_rewards.append(ep_rew)
        ep_lens.append(ep_len)
        ep_infos.append(info)
        if verbose:
            print("Finished Episode", ep_index + 1, "Reward:", ep_rew, "Length:", ep_len)
    
    print("Completed Eval of", num_ep, "Episodes")
    print("Avg. Reward:", np.mean(ep_rewards), "Avg. Length", np.mean(ep_lens))
    return np.mean(ep_rewards), frames

def test(name, num_ep=10, deterministic=True, verbose=1, gif=False):
    model, env = load_from_name(name, load_env=True)
    _, frames = eval_policy(model, env, num_ep=num_ep, deterministic=deterministic, verbose=verbose, gif=gif, render=True)
    if gif:
        import imageio
        if name.endswith('/'):
            name = name[:-1]
        if name.startswith(BASE):
            # Remove the base
            name = name[len(BASE):]
            if name.startswith('/'):
                name = name[1:]
        render_path = os.path.join(RENDERS, name + '.gif')
        print("Saving gif to", render_path)
        os.makedirs(os.path.dirname(render_path), exist_ok=True)
        imageio.mimsave(render_path, frames[::5], subrectangles=True, duration=0.06)
    env.close()
    del model

def test_composition(low_level_path, high_level_path, num_ep=100, deterministic=True, gif=False, render=False):
    # Test compositions from low levels and high levels
    if not low_level_path.startswith('/'):
        low_level_path = os.path.join(LOW_LEVELS, low_level_path)
    if not high_level_path.startswith('/'):
        high_level_path = os.path.join(HIGH_LEVELS, high_level_path)
    
    high_level_env_args = [
        "rand_init",
    ]

    # Now, combine the parameters.
    low_level_params = ModelParams.load(low_level_path)
    high_level_params = ModelParams.load(high_level_path)
    # Take the task from the high level, Env Args from low level.
    task = high_level_params['env'].split('_')[0]
    agent = low_level_params['env'].split('_')[1]

    combined_params = copy.deepcopy(high_level_params)
    combined_params['env_args'] = copy.deepcopy(low_level_params['env_args'])
    for param in high_level_env_args:
        combined_params['env_args'][param] = high_level_params['env_args'][param]
    combined_params['env_wrapper_args']['low_level'] = low_level_path
    combined_params['env'] = task + '_' + agent
    model, env = load(high_level_path, combined_params, best=True, load_env=True)

    verbose = 1 if render else 0
    avg_reward, frames = eval_policy(model, env, num_ep=num_ep, deterministic=deterministic, verbose=verbose, gif=gif, render=render)
    
    if gif:
        import imageio
        render_path = os.path.join(RENDERS, 'composition_test' + '.gif')
        print("Saving gif to", render_path)
        os.makedirs(os.path.dirname(render_path), exist_ok=True)
        imageio.mimsave(render_path, frames[::1], subrectangles=True, duration=0.05)
    
    return avg_reward
    