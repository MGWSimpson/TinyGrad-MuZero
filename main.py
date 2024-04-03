
from core.train import  train
from config.classic_control.__init__ import ClassicControlConfig
import argparse

import ray

def main():


    parser = argparse.ArgumentParser(description='MuZero Pytorch Implementation')
    parser.add_argument('--env', default="LunarLander-v2", help='Name of the environment')
    parser.add_argument('--num_actors', type=int, default=2,
                        help='Number of actors running concurrently (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed (default: %(default)s)')


    args = parser.parse_args()

    config = ClassicControlConfig(args=args)
    ray.init()
    train(config)




if __name__== "__main__":
    main()




