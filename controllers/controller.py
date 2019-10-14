from tkinter import *
from view.gui import Window
from models.eventgen import EventGen, ce_str
import logging

class Controller:
    def __init__(self, args):
        self.args = args
        self.eventgen = EventGen(7,7,"uniform", 10, 3, 1, logging.getLogger(''))


    def update_base_station(self, parameter_list):
        raise NotImplementedError

    def train(self):
        print("train")

    def start(self):
        if self.args.display:
            print("verbosity turned on")
            root = Tk()
            #size of the window
            root.geometry("425x325")
            app = Window(root, self)
            app.mainloop()
        else:
            print("sssssss")
            self.a = 0
            self.train()

    def pop_event(self):
        return self.eventgen.pop()

    def event_new(self, t, cell, dt=None):
        self.eventgen.event_new(t, cell, dt)

        