from controllers.runner import MultiChannelPPORunner, SingleChannelRunner, MultiChannelRunner, MultiChannelRandom
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
        # runner.test()
    elif args.model == "random":
        runner = MultiChannelRandom()
        runner.run()

    # elif args.model == "LDS":
    #     runner = MultiChannelLDSRunner(args)
    #     runner.train()

    elif args.channel == "single":
        runner = SingleChannelRunner(args)
        runner.train()

    elif args.channel == "multi":
        runner = MultiChannelRunner(args)
        runner.train()


