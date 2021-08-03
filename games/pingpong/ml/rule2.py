"""
The template of the script for the machine learning process in game pingpong
"""

import pickle
import os
import math
import random


class MLPlay:
    def __init__(self, side):
        """
        Constructor

        @param side A string "1P" or "2P" indicates that the `MLPlay` is used by
               which side.
        """
        self.ball_served = False
        self.side = side
        self.ball_last_x = 98
        self.ball_last_y = 415
        self.block_x = 0
        self.block_speed = 0
        self.predict_x = 0
        self.last_speed_x = 7
        self.isPredict = False
        self.count = 0

    def update(self, scene_info):
        """
        Generate the command according to the received scene information
        """
        if scene_info["status"] != "GAME_ALIVE":
            # print("ball_x=", scene_info.get("ball")[0])
            print("ball_speend=", scene_info.get("ball_speed")[0])
            return "RESET"

        if not self.ball_served:
            self.ball_served = True
            # print(scene_info.get('blocker')[0])
            return "SERVE_TO_LEFT"
        else:
            self.count += 1
            command = ""
            ball_x = (scene_info.get('ball')[0])
            ball_y = (scene_info.get('ball')[1])
            p2_x = (scene_info.get('platform_2P')[0])
            if(scene_info.get('blocker')[0]-self.block_x > 0):
                self.block_speed = 5
            else:
                self.block_speed = -5
            self.block_x = scene_info.get('blocker')[0]
            speed_y = scene_info.get('ball_speed')[1]
            speed_x = scene_info.get('ball_speed')[0]
            if(self.ball_last_y >= 415 or self.ball_last_y <= 80):
                self.isPredict = False
            self.predict_x = self.P2movePredict(
                ball_x, ball_y, speed_x, speed_y, self.block_x)
            # if(not self.isPredict):
            #print("final predict=", self.predict_x)
            self.isPredict = True
            #print(ball_x, ball_y)
            # print(self.predict_x)
            if(self.predict_x != -1):

                if(p2_x > self.predict_x-random.randint(10, 20)):
                    command = "MOVE_LEFT"
                elif(p2_x < self.predict_x-random.randint(30, 40)):
                    command = "MOVE_RIGHT"
                else:
                    command = "NONE"
            else:
                if(p2_x > 100):
                    command = "MOVE_LEFT"
                elif(p2_x < 90):
                    command = "MOVE_RIGHT"
                else:
                    command = "NONE"
            self.ball_last_x = ball_x
            self.ball_last_y = ball_y
            self.last_speed_x = speed_x
            # print(scene_info.get('ball_speed'))
            return command

    def P2movePredict(self, x, y, speed_x, speed_y, block_x):
        block_speed = self.block_speed
        count = self.count-1
        # print("========================")
        while(True):
            # if(not self.isPredict):
            #print("P1 predict=", x, y, speed_x, speed_y, block_x)
            count += 1
            if(count % 100 == 0):
                if(speed_x < 0):
                    speed_x -= 1
                else:
                    speed_x += 1
                if(speed_y < 0):
                    speed_y -= 1
                else:
                    speed_y += 1
            if(block_x <= 0):
                block_x = 0
                block_speed = -block_speed
            if(block_x >= 170):
                block_x = 170
                block_speed = -block_speed
            # if x==195 or x==0 the speed already change
            # if x==195 or x==0 the speed already change
            if(x+speed_x <= 0):
                x = 0
                y += speed_y
                speed_x = -speed_x
                block_x += block_speed
                continue
            if(x+speed_x >= 195):
                x = 195
                y += speed_y
                speed_x = -speed_x
                block_x += block_speed
                continue
            # move up
            if(speed_y < 0):
                dy = y-80

                if(dy <= abs(speed_y)):
                    x = x+abs((y-80)/speed_y)*speed_x
                    y += speed_y
                    break
                next_x = x+speed_x
                d_block_x = block_x+block_speed
                # ball collide the block side
                if(y+speed_y <= 260 and y+5+speed_y >= 240):
                    if(speed_x > 0 and next_x+5 >= d_block_x and next_x <= d_block_x+30):
                        x = d_block_x-5
                        y += speed_y
                        speed_x = -speed_x
                        block_x += block_speed
                        # print("collide")
                        continue
                    elif(speed_x < 0 and next_x <= d_block_x+30 and next_x+5 >= d_block_x):
                        x = d_block_x+30
                        y += speed_y
                        speed_x = -speed_x
                        block_x += block_speed
                        # print("collide")
                        continue
                        # move up
            # move down
            else:
                # consider that the block will collide the blocker
                # if not just return -1 when the ball went over the blocker
                if(y >= 260):
                    #print("----------------run over-------------------------")
                    return -1
                if(y+5 >= 240):
                    speed_y = -speed_y
                    y = 235+speed_y
                    x += speed_x
                    block_x += block_speed
                    continue
            x += speed_x
            y += speed_y
            block_x += block_speed
        #   finally normalize
        if(x < 0):
            x = 0
        elif x > 195:
            x = 195
        # print("predict x=", x)
        return x

    def reset(self):
        """
        Reset the status
        """
        self.ball_served = False
