from tkinter import *
from gui import Window
import argparse
from simulation import Simulation



def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")

    parser.add_argument("--display", action="store_true", default=False)


if __name__ == "__main__":

    
    root = Tk()
    #size of the window
    root.geometry("425x325")

    app = Window(root)
    root.mainloop()