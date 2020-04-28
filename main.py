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
    if args.model.upper() == "PPO":
        runner = DCARunner(args)
        if args.test:
            runner.test()
        else:
            runner.train()
    elif args.model.upper() == "A2C":
        runner = DCARunner(args)
        if args.test:
            runner.test()
        else:
            runner.train()
    elif args.model.upper() == "DQN":
        runner = DCARunner(args)
        if args.test:
            runner.test()
        else:
            runner.train()
    elif args.model.upper() == "ACER":
        runner = DCARunner(args)
        if args.test:
            runner.test()
        else:
            runner.train()

    elif args.model.upper() == "RANDOM":
        runner = DCARunner(args)
        runner.test()

    elif args.model.upper() == "DCA":
        runner = DCARunner(args)
        runner.test()  
    else:
        print("something wrong")



