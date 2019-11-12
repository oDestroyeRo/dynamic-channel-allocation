from controllers.runner import SingleChannelRunner, MultiChannelRunner, MultiChannelPPORunner
import argparse

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--channel", default="single")
    parser.add_argument("--agent", default="single")
    parser.add_argument("--model", default="dqn")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    if args.model == "ppo":
        runner = MultiChannelPPORunner(args)
        runner.train()
        runner.test()

    elif args.channel == "single":
        runner = SingleChannelRunner(args)
        runner.train()

    elif args.channel == "multi":
        runner = MultiChannelRunner(args)
        runner.train()

