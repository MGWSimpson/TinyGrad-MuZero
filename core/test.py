


from tinygrad import Tensor
import os


from .mcts import MCTS, Node
from .utils import select_action
import multiprocessing


def _test(config, model, ep_i, device, render, save_video, save_path):
   
    env = config.new_game(save_video=save_video, save_path=save_path,
                              episode_trigger=lambda episode_id: True, uid=ep_i)
    done = False
    ep_reward = 0
    obs = env.reset()
    while not done:
        if render:
            env.render()
        root = Node(0)
        obs = Tensor(obs).unsqueeze(0)
        root.expand(env.to_play(), env.legal_actions(), model.initial_inference(obs))
        MCTS(config).run(root, env.action_history(), model)
        action, _ = select_action(root, temperature=1, deterministic=True)
        obs, reward, done, info = env.step(action.index)
        ep_reward += reward
       
    env.close()

    return ep_reward

"""
TODO: should rewrite to make multi threaded but will work for now
"""
def test(config, model, episodes, device, render, save_video=False):
    model.eval()
    save_path = os.path.join(config.exp_path, 'recordings')


    test_reward = 0 
    for ep_i in range(2):
        test_reward += _test(config, model, ep_i,'cpu', render, save_video, save_path)

 
    return test_reward / episodes
