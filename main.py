from controllers.runner import SingleChannelRunner, MultiChannelRunner
import argparse

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--channel", default="single")
    parser.add_argument("--agent", default="single")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    if args.channel == "single":
        runner = SingleChannelRunner(args)
        runner.train()

    if args.channel == "multi":
        runner = MultiChannelRunner(args)
        runner.train()

