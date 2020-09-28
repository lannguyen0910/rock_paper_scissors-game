import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.autograd import Variable
import numpy as np
import os
import random
import cv2
#device = torch.device('cpu')
resnet = models.resnet50(pretrained=True)

for params in resnet.parameters():
    params.requires_grad = False

in_features = resnet.fc.in_features
resnet.fc = nn.Linear(in_features=in_features, out_features=3)
resnet.load_state_dict(torch.load('model2.pt', map_location=torch.device('cpu')))


IMG_SIZE = 128
options = ['rock', 'paper', 'scissors']
winRule = {'rock': 'scissors', 'scissors': 'paper', 'paper': 'rock'}
rounds = 0
botScore = 0
playerScore = 0


def prepImg(path):
    return cv2.resize(path, (IMG_SIZE, IMG_SIZE)).reshape(1, IMG_SIZE, IMG_SIZE, 3)


def updateScore(play, bplay, p, b):
    winRule = {'rock': 'scissors', 'scissors': 'paper', 'paper': 'rock'}
    if play == bplay:
        return p, b
    elif bplay == winRule[play]:
        return p+1, b
    else:
        return p, b+1

def inv_normalize(img):
    mean = torch.Tensor([0.4914, 0.4822, 0.4465]).unsqueeze(-1)
    std= torch.Tensor([0.2023, 0.1994, 0.2010]).unsqueeze(-1)
    img = (img.view(3, -1) * std + mean).view(img.shape)
    img = img.clamp(0, 1)
    return img

def predict_images(model, images):
    model.eval()

    with torch.no_grad():
     
        image = torch.from_numpy(images)
        image = inv_normalize(image)
        test = Variable(image).view(-1, 3, IMG_SIZE, IMG_SIZE)
        #test = test.to(device)

        outputs = model(test)
        preds = F.softmax(outputs.data, dim=1)
        print('Preds: ', preds)

    return preds.cpu().numpy()[0]


cap = cv2.VideoCapture(0)
ret, frame = cap.read()

NUM_ROUNDS = 3
bplay = ""
labels_dict = {0: 'paper', 1: 'rock', 2: 'scissors'}

while True:
    ret, frame = cap.read()
#     if ret:
#         assert not isinstance(frame, type(None)), 'frame not found'
    # print(ret)
    frame = cv2.putText(frame, "Press Space to start", (160, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
    cv2.imshow('Rock paper scissors', frame)
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
                resnet, prepImg(frame[100: 300, 150: 350])))
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
    frame = cv2.putText(frame, "Player : {}      Bot : {}".format(
        playerScore, botScore), (120, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
    cv2.imshow('Rock Paper Scissor', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
