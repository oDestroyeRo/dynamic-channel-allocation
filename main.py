from view.gui import Window
import argparse
from controllers.simulation import Simulation
from controllers.controller import Controller



def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--display", action="store_true", default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    c = Controller(args)
    c.start()

    # root.update()ˇ