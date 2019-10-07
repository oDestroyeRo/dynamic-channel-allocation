from tkinter import *
from gui import Window
import argparse



def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")

    parser.add_argument("--display", action="store_true", default=False)


if __name__ == "__main__":
    root = Tk()
    #size of the window
    root.geometry("400x300")

    app = Window(root)
    root.mainloop()