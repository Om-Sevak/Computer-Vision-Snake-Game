import math

import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)

class SnakeClass:

    def __init__(self, foodPath):
        self.nodes = []  #nodes of snake
        self.length = [] # distance between each node
        self.currLength = 0 #current length of snake
        self.maxLength = 500 #max allowed length of snake
        self.prevNode = 0, 0
        self.score = 0
        self.foodImg = cv2.imread(foodPath, cv2.IMREAD_UNCHANGED)
        self.foodImg = cv2.resize(self.foodImg, (50, 50))
        self.foodHeight, self.foodWidth, _ = self.foodImg.shape
        self.foodCord = 0, 0
        self.randFoodGenerator()
        self.over = False

    def randFoodGenerator(self):
        x = np.random.randint(0, 1280 - self.foodWidth)
        y = np.random.randint(0, 720 - self.foodHeight)
        self.foodCord = x, y
        return x, y


    def update(self, img, currHead):

        if self.over:
            cvzone.putTextRect(img, "Game Over", [300,400], scale=7, thickness=5, offset=20)
            cvzone.putTextRect(img, f'Score: {self.score}' , [300, 550], scale=7, thickness=5, offset=20)
            self.score = 0

        else:
            prevX, prevY = self.prevNode
            currX, currY = currHead

            #adding this points to the corresponding list
            self.nodes.append([currX, currY])
            distance = math.hypot(currX-prevX, currY - prevY)
            self.length.append(distance)
            self.currLength += distance
            self.prevNode = currX, currY

            #length reduction
            if self.currLength > self.maxLength:
                for i, len in enumerate(self.length):
                    self.currLength -= len
                    self.length.pop(i)
                    self.nodes.pop(i)

                    if self.currLength < self.maxLength:
                        break

            #check if snake has eaten food
            rx, ry = self.foodCord
            if rx <= currX <= rx + self.foodWidth and ry <= currY <= ry + self.foodHeight:
                self.randFoodGenerator()
                self.maxLength += 100
                self.score += 1


            #Drawing snake using updated values
            if self.nodes:
                for i, node in enumerate(self.nodes):
                    if i > 0:
                        cv2.line(img, self.nodes[i-1],self.nodes[i], (0,0,169), 20)
                cv2.circle(img, self.nodes[-1], 20, (200, 0, 200), cv2.FILLED)
            #Drawing food
            img = cvzone.overlayPNG(img, self.foodImg, self.foodCord)

            # check if snake has hit itself
            pts = np.array(self.nodes[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, (0, 255, 0), thickness=2)
            if -1 <= cv2.pointPolygonTest(pts, (currX, currY), True) <= 1:
                self.nodes = []  # nodes of snake
                self.length = []  # distance between each node
                self.currLength = 0  # current length of snake
                self.maxLength = 500  # max allowed length of snake
                self.prevNode = 0, 0
                self.randFoodGenerator()
                self.over = True
        return img


game = SnakeClass("Donut.png")

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    numHands, img = detector.findHands(img, flipType= False)

    if numHands:
        lmList = numHands[0]['lmList']
        pointIndex = lmList[8][0:2]
        img = game.update(img, pointIndex)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

