import torch
import torch.nn as nn
import cv2
import streamlit as st
import numpy as np
import asyncio
import edge_tts
import pygame
a=False
b=1
c=0
sk=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
st.title("Real-Time Handwritten Number Detection")
st.subheader('Enter one digit at a time in camera view Write the number big with thiker pen')
cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
indi=cv2.VideoCapture("Speak_Indicator.webm")
framewindow=st.empty()
write_window=st.empty()
device=torch.device("cuda"if torch.cuda.is_available() else"cpu")
print("Using device:",device)
async def speak(text):
    communicate=edge_tts.Communicate(text,"en-US-AriaNeural")
    await communicate.save("speak.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load("speak.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        framewindow.image(frame0,width=100)
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()
class my_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(1,8,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(8,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer0=nn.Sequential(
            nn.Linear(16*25*25,5000),
            nn.ReLU(),
            nn.Linear(5000,1000),
            nn.ReLU(),
            nn.Linear(1000,500),
            nn.ReLU(),
            nn.Linear(500,100),
            nn.ReLU(),
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Linear(50,10))
    def forward(self,x):
        x=self.layer(x)
        x=torch.flatten(x,1)
        x=self.layer0(x)
        return x
model=my_model().to(device)
model.load_state_dict(torch.load("Number_Detection.pth", map_location=device))
model.eval()
c1,c2=st.columns(2)
with c1:
    if st.button("Start"):
        a=True
with c2:
    if st.button("Stop"):
        a=False
while a:
    ret,frame=cam.read()
    ret0,frame0=indi.read()
    if not ret0:
        indi.set(cv2.CAP_PROP_POS_FRAMES,0)
        continue
    if not ret:
        print("Failed to grab frame")
        break
    image=cv2.resize(frame,(100,100))
    display=cv2.rectangle(image,(2,2),(98,98),(190,25,90),2)
    frame=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,frame = cv2.threshold(frame,100,255,cv2.THRESH_BINARY_INV)
    frame=cv2.filter2D(frame,-1,sk)
    frame=frame/255.0
    frame = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        output=model(frame)
        pred=output.argmax(dim=1).item()
    framewindow.image(display,width=100)
    print(pred)
    write_window.write(f'Predicted Number: {pred}')
    if b==pred:
        c=c+1
    else:
        c=0
    b=pred
    if c==81:
        asyncio.run(speak(f'Predicted Number: {pred}'))
        c=0
cam.release()
indi.release()