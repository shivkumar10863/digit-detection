import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from torchvision.datasets import MNIST as mn
from torchvision import transforms
from torch.utils.data import DataLoader as dl
transform=transforms.Compose([
    transforms.Resize((100,100)),
    transforms.RandomAffine(degrees=15,translate=(0.2, 0.2),scale=(0.7, 1.3)),
    transforms.ColorJitter(brightness=0.4,contrast=0.4),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3)
])
device=torch.device("cuda"if torch.cuda.is_available() else"cpu")
print("Using device:",device)
x_tr=mn(
    root='data',
    train=True,
    download=True,
    transform=transform)
x_te=mn(
    root='data',
    train=False,
    download=True,
    transform=transform)
tr_l=dl(x_tr,batch_size=32,shuffle=True)
te_l=dl(x_te,batch_size=32,shuffle=False) 
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
loss_fn=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
epochs=3
for epoch in range(epochs):
    total_loss=0
    model.train()
    for image,lebel in tr_l:
        image=image.to(device)
        lebel=lebel.to(device)
        optimizer.zero_grad()
        y_lebel=model(image)
        loss=loss_fn(y_lebel,lebel)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(tr_l)}")
torch.save(model.state_dict(),'Number_Detection.pth')
model.eval()
correct=0
total=0
with torch.no_grad():
    for image,target in te_l:
        image=image.to(device)
        target=target.to(device)
        y_p=model(image)
        predicted=y_p.argmax(dim=1)
        total+=target.size(0)
        correct+=(predicted==target).sum().item()
accuracy=100*correct/total
print(f"Test Accuracy: {accuracy:.2f}%")
index=800
image,target=x_te[index]
img=(image.squeeze().cpu().numpy() * 255).astype(np.uint8)
img=cv2.resize(img,(1400,700))
with torch.no_grad():
        image=image.unsqueeze(0).to(device)  
        label=model(image)
        predicted_label=label.argmax(dim=1).item()
img=cv2.putText(img,f"Predicted Label: {predicted_label}  True Label: {target}",(50,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("index:", index)
print(f"True Label: {target}")
print(f"Predicted Label: {predicted_label}")
