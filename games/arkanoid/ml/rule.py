"""
The template of the main script of the machine learning process
"""
import pickle
import os
import math
import random


class MLPlay:
    def __init__(self):
        """
        Constructor
        """
        self.ball_served = False
        self.last_x = 200
        self.last_y = 400
        self.predict_x = 0
        self.slope = -1

    def update(self, scene_info):
        """
        Generate the command according to the received `scene_info`.
        """
        # Make the caller to invoke `reset()` for the next round.
        if (scene_info["status"] == "GAME_OVER" or
                scene_info["status"] == "GAME_PASS"):
            # print(self.last_x)
            return "RESET"

        if not self.ball_served:
            self.ball_served = True
            command = "SERVE_TO_RIGHT"
        else:
            command = ""
            x = (scene_info.get('ball')[0])
            y = (scene_info.get('ball')[1])
            now_plat_x = (scene_info.get('platform')[0])
            # print(now_plat_x)
            # print(scene_info.get('platform'))
            # initial
            self.slope = (y-self.last_y)/(x-self.last_x)
            # print(self.slope)
            # print(self.predict_x)
            if(y-self.last_y > 0 and y > 150):
                # if(self.predict_x < 0):
                self.predictPoint(x, y)
                # print("predict="+str(self.predict_x))
                if(now_plat_x > self.predict_x - random.randint(0, 20)):
                    command = "MOVE_LEFT"
                elif(now_plat_x < self.predict_x-random.randint(30, 50)):
                    command = "MOVE_RIGHT"
                else:
                    command = "NONE"
            else:
                self.predict_x = -100
                if(now_plat_x > 140):
                    command = "MOVE_LEFT"
                elif(now_plat_x < 60):
                    command = "MOVE_RIGHT"
                else:
                    command = "NONE"
            self.last_x = x
            self.last_y = y

        return command

    def predictPoint(self, x, y):
        while True:
            if(self.slope > 0):
                dx = 200-x
                dy = abs(dx*self.slope)
            else:
                dx = x
                dy = abs(dx*self.slope)
            if(y+dy >= 400):
                break
            else:
                # rebound
                y += dy
                if (self.slope > 0):
                    x = 200
                else:
                    x = 0
                self.slope = -self.slope
            # print("-----------")
        self.predict_x = (400-y)/self.slope+x
        # print(self.predict_x)

    def reset(self):
        """
        Reset the status
        """
        self.ball_served = False
