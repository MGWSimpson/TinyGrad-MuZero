
from core.train import  train
from core.test import test
from config.classic_control.__init__ import ClassicControlConfig
import argparse
import os
import ray
import logging.config
from core.utils import init_logger, make_results_dir


from tinygrad import nn

def main():


    parser = argparse.ArgumentParser(description='MuZero Pytorch Implementation')
    parser.add_argument('--env', default="CartPole-v1", help='Name of the environment')
    parser.add_argument('--num_actors', type=int, default=1,
                        help='Number of actors running concurrently (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed (default: %(default)s)')
    parser.add_argument('--opr', default= "train")


    args = parser.parse_args()

    config = ClassicControlConfig(args=args)


    exp_path = './experiments/'

    exp_path, log_base_path = make_results_dir(exp_path, args)

    init_logger(log_base_path)




    try:
        if args.opr == "train":
            ray.init()
            train(config)
            ray.shutdown()
        elif args.opr == "test": 
            assert os.path.exists(config.model_path), 'model not found at {}'.format(config.model_path)
            model = config.get_uniform_network()
            model.load_state_dict(nn.state.safe_load(config.model_path))
            test_score = test(config, model, args.test_episodes, device='cpu', render=args.render,
                              save_video=True)
        else:
            raise Exception('Please select a valid operation (--opr) to be performed')
    except Exception as e: 
        print(e) 




if __name__== "__main__":
    main()




