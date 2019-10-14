from tkinter import *
import numpy as np
from models.eventgen import CEvent

class Window(Frame):
    def __init__(self, master, controller):
        Frame.__init__(self, master)
        self.controller = controller
        self.master = master
        self.shape = Canvas(master)
        self.base_stations = np.zeros(shape=(7,7),dtype=int)
        self.cevent = None
        self.init_window()

    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget      
        self.master.title("GUI")

        # allowing the widget to take the full space of the root window
        # self.pack(fill=BOTH, expand=1)
        self.shape.pack(fill=BOTH, expand=1)

        # creating a button instance
        # quitButton = Button(self, text="Quit", command=self.client_exit)

        x=30
        y=10
        # print(self.eventgen.event_new())
        dt = 0
        for i in range(0, 7):
            x = 30 + i * 20
            for j in range(0,7):
                points = [x,y,x+20,y+13,x+20,y+40,x+0,y+53,x-20,y+40,x-20,y+13,x+0,y+0]
                self.base_stations[i][j] = self.shape.create_polygon(points, outline="#476042", fill='gray', width=2)

                self.shape.create_text(x ,y+10, text=str(i)+","+str(j))
                x = x + 40
                # self.eventgen.event_new(0, (i, j), dt)
                self.controller.event_new(0, (i, j), dt)
                cevent = self.controller.pop_event()
                if (cevent[1] == CEvent.NEW):
                    # print("NEW")
                    self.shape.itemconfig(self.base_stations[cevent[2][0]][cevent[2][1]], fill='red')
                elif(cevent[1] == CEvent.HOFF):
                    self.shape.itemconfig(self.base_stations[cevent[2][0]][cevent[2][1]], fill='blue')
                dt += 1
            y=y+40

        # self.shape.itemconfig(self.base_stations[0][1], fill='blue')
        # self.shape.itemconfig(self.base_stations[0][2], fill='gray')
            
        # self.shape.itemconfig(self.base_stations[1][1], fill='red')
        # self.shape.create_line(0, 0, 200, 100)

        # placing the button on my window
        # quitButton.place(x=0, y=0)

    def client_exit(self):
        exit()





    def update_base_station(self, color, rol, col):
        self.shape.itemconfig(self.base_stations[rol][col], fill=color)
