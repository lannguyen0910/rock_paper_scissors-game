import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.autograd import Variable
import numpy as np
import pandas as pd
import os
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

resnet = models.resnet50(pretrained=True)
resnet = resnet.to(device)

for params in resnet.parameters():
    params.requires_grad = False

in_features = resnet.fc.in_features
resnet.fc = nn.Linear(in_features = in_features, out_features=3).to(device)

resnet.load_state_dict(torch.load('model1.pt'))


IMG_SIZE = 128
options = ['rock', 'paper', 'scissor']
winRule = {'rock': 'scissor', 'scissor': 'paper', 'paper': 'rock'}
rounds = 0
botScore = 0
playerScore = 0


def prepImg(path):
    return cv2.resize(path, (IMG_SIZE, IMG_SIZE)).reshape(1, IMG_SIZE, IMG_SIZE, 3)


def updateScore(play, bplay, p, b):
    winRule = {'rock': 'scissor', 'scissor': 'paper', 'paper': 'rock'}
    if play == bplay:
        return p, b
    elif bplay == winRule[play]:
        return p+1, b
    else:
        return p, b+1


def predict_images(model, images):
    model.eval()

    with torch.no_grad():
        test = Variable(images).view(-1, 3, IMG_SIZE, IMG_SIZE)
        test = test.to(device)

        outputs = model(test)
        preds = F.softmax(outputs.data, dim=1)

    return preds.cpu().numpy()[0]


cap = cv2.VideoCapture(0)
ret, frame = cap.read()

NUM_ROUNDS = 3
bplay = ""


while True:
    ret, frame = cap.read()
    if ret:
        assert isinstance(frame, type(None)), 'frame not found'
    # print(ret)
    frame = cv2.putText(frame, "Press Space to start", (160, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
    cv2.imshow('Rock Paper Scissors',frame)
    if cv2.waitKey(1) & 0xff == ord(' '):
        break

for rounds in range(NUM_ROUNDS):
    pred = ""
    for i in range(90):
        ret, frame = cap.read()

        # Countdown
        if i//20 < 3:
            frame = cv2.putText(frame, str(
                i//20+1), (320, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (250, 250, 0), 2, cv2.LINE_AA)

        # Prediction
        elif i/20 < 3.5:
            pred = np.argmax(predict_images(
                resnet, prepImg(frame[0:200, 0:200])))
            pred = labels_dict[pred]
        # Get Bots Move
        elif i/20 == 3.5:
            bplay = random.choice(options)
            print(pred, bplay)

        # Update Score
        elif i//20 == 4:
            playerScore, botScore = updateScore(
                pred, bplay, playerScore, botScore)
            break

        cv2.rectangle(frame, (100, 150), (300, 350), (255, 255, 255), 2)
        frame = cv2.putText(frame, "Player : {}      Bot : {}".format(
            playerScore, botScore), (120, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, pred, (150, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "Bot Played : {}".format(
            bplay), (300, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
        cv2.imshow('Rock Paper Scissor', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

if playerScore > botScore:
    winner = "You Won :)"
elif playerScore == botScore:
    winner = "Its a Tie"
else:
    winner = "AI Won :("

while True:
    ret, frame = cap.read()
    frame = cv2.putText(frame, winner, (230, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "Press q to quit", (190, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "Player : {}      AI : {}".format(
        playerScore, botScore), (120, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
    cv2.imshow('Rock Paper Scissor', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
