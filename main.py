from controllers.runner import DCARunner
import argparse

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--channel", default="single")
    parser.add_argument("--agent", default="single")
    parser.add_argument("--model", default="ppo")
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    if args.model == "ppo":
        runner = DCARunner(args)
        if args.test:
            runner.test()
        else:
            runner.train()
    elif args.model == "a2c":
        runner = DCARunner(args)
        if args.test:
            runner.test()
        else:
            runner.train()
    elif args.model == "dqn":
        runner = DCARunner(args)
        if args.test:
            runner.test()
        else:
            runner.train()

    elif args.model == "random":
        runner = DCARunner(args)
        runner.test()



