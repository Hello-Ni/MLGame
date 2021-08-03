import pickle
import os
import numpy as np
import random
from numpy.core.defchararray import count
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn import svm


def openPickle():
    ball_x = []
    ball_y = []
    block_x = []
    ball_speed_x = []
    ball_speed_y = []
    direction = []
    platform = []
    my_command = []
    count = 0
    zero = 0
    one = 0
    two = 0
    for j in range(101):
        file_path = "../log/"+str(j)+".pickle"
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        # print(str(data['ml_1P']['scene_info'][0])+"\n")
        scene_info = data['ml_1P']['scene_info']
        #command = data['ml_1P']['command']
        # filteData(s['ball'][0], s['ball'][1])
        for i, s in enumerate(scene_info[1:]):
            count += 1
            x = (s['ball'][0])
            y = (s['ball'][1])
            b_x = s['blocker'][0]
            p1_x = (s['platform_1P'][0])
            # print(p1_x)
            if(scene_info[i-1]['blocker'][0]-b_x > 0):
                block_speed = 5
            else:
                block_speed = -5

            speed_y = s['ball_speed'][1]
            speed_x = s['ball_speed'][0]
            # if command wrong return true to remove command(not the efficient data)
            predict_x = filtData(p1_x, x, y, speed_x,
                                 speed_y, block_speed, b_x, count)
            if(predict_x != -1):
                # print(self.predict_x)
                if(p1_x > predict_x - random.randint(10, 20)):
                    command = 1
                    one += 1
                elif(p1_x < predict_x-random.randint(30, 40)):
                    command = 2
                    two += 1
                else:
                    command = 0
                    zero += 1
                # print(scene_info.get('ball_speed'))
            else:
                if(p1_x > 100):
                    command = 1
                    one += 1
                elif(p1_x < 90):
                    command = 2
                    two += 1
                else:
                    command = 0
                    zero += 1
            my_command.append(command)
            ball_x.append(x)
            ball_y.append(y)
            block_x.append(b_x)
            platform.append(p1_x)
            ball_speed_x.append(speed_x)
            ball_speed_y.append(speed_y)
        count = 0
    numpy_data = np.array(
        [ball_x, ball_y, block_x, ball_speed_x, ball_speed_y,  platform])
    # print(my_command)
    print(zero, one, two)
    print(len(numpy_data[2]))
    trainData(np.transpose(numpy_data), my_command)
    # KMeanTrain(np.transpose(numpy_data), my_command)


def filtData(p1_x, x, y, speed_x, speed_y, block_speed, block_x, c):
    count = c-1
    while(True):
        # if(not self.isPredict):
        # print("P1 predict=", x, y, speed_x, speed_y, block_x)
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
        if(x+speed_x <= 0):
            x = 0
            y += speed_y
            block_x += block_speed
            speed_x = -speed_x
            continue
        if(x+speed_x >= 195):
            x = 195
            y += speed_y
            block_x += block_speed
            speed_x = -speed_x
            continue
        # move down
        if(speed_y > 0):
            dy = 415-y
            slope = speed_y/speed_x
            if(dy <= speed_y):
                x = x+abs((415-y)/speed_y)*speed_x
                y += speed_y
                break
            next_x = x+speed_x
            d_block_x = block_x+block_speed
            # ball collide the block side
            if(y+speed_y <= 260 and y+5+speed_y >= 240):
                if(speed_x > 0 and next_x+5 >= d_block_x and next_x <= d_block_x+30 and y+speed_y <= 260):
                    x = d_block_x-5
                    y += speed_y
                    speed_x = -speed_x
                    block_x += block_speed
                    # print("collide")
                    continue
                elif(speed_x < 0 and next_x <= d_block_x+30 and next_x+5 >= d_block_x and y+speed_y <= 260):
                    x = d_block_x+30
                    y += speed_y
                    speed_x = -speed_x
                    block_x += block_speed
                    # print("collide")
                    continue
                    # move up
        else:
            # consider that the block will collide the blocker
            # if not just return -1 when the ball went over the blocker
            if(y <= 240):
                # print("----------------run over-------------------------")
                return -1
            if(y <= 260):
                speed_y = -speed_y
                y = 260+speed_y
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

    return x


def trainData(data, command):
    # split the train and test data
    print("train")
    data_train, data_test, cmd_train, cmd_test = train_test_split(
        data, command, test_size=0.1, random_state=9)
    # model = svm.SVC(decision_function_shape='ovr')
    # model.fit(data_train, cmd_train)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(data_train, cmd_train)
    predict_y = model.predict(data_test)
    print(model.score(data_test, cmd_test))
    print("train finish")
    # pca = PCA(n_components=2).fit(data_test)
    # pca_2d = pca.transform(data_test)
    # c2 = 0
    # for i in range(0, pca_2d.shape[0]):
    #     if predict_y[i] == 0:
    #         c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
    #     elif predict_y[i] == 1:
    #         c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
    #     elif predict_y[i] == 2:
    #         c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')

    # print(c1, c2, c3)
    # plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2', 'Cluster 3'])
    # plt.show()
    with open("1Pmodel.pickle", 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    openPickle()
