
import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
import gzip



batch = 34
batchTest = 21

def PreProcessing(f):
    data = []
    dataLabels = []
    for line in f:
        x = line.decode().strip()
        x = x.split(',')
        temp = []
        if x[len(x)-1]== 'normal.':
            x[len(x)-1] = '1.0'
        else :
            x[len(x)-1] = '0.0'
        for i in x:
            if i[0].isdigit():
                temp.append(float(i))
            elif i.isdecimal():
                temp.append(float(i))
            else:
                temp2 = 0;
                for j in i:
                    temp2 = temp2 + ord(j)
                temp.append(temp2)
        dataLabels.append(int(temp[-1]))
        temp = temp[:-1]
        data.append(temp)
    return data, dataLabels
    



class NeuralNetwork(nn.Module):
    def __init__(self):
       super(NeuralNetwork, self).__init__()
    
       self.linear1 = nn.Linear(41,25).double()
       self.relu1 = nn.ReLU()
       self.linear2 = nn.Linear(25,10).double()
       self.relu2 = nn.ReLU()
       self.linear3 = nn.Linear(10,2).double()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
     
        return x

model = NeuralNetwork()

def train_loop(trainingNu,trainingLabelNu, model, loss_fn, optimizer):
    for i in range(0,len(trainingNu),batch):
        res = torch.empty((batch,2))
        label = torch.empty(batch, dtype=torch.long)
        for j in range(0,batch):
            if (j+i<len(trainingNu)):
                pred = model(trainingNu[j+i])
                res[j] = pred
                label[j] = trainingLabelNu[j+i]
            else:
                continue
      
        loss = loss_fn(res, label)
        # Backpropagation - 'timwria' neurwnikou. DEN allazoun autew oi grammes
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
             
            
def test_loop(testNu,testLabelNu, model, loss_fn):
    test_loss, correct = 0, 0
    with torch.no_grad():
        y = 0 
        for i in range(0,len(testNu),batchTest):
            res = torch.empty((batchTest,2))
            label = torch.empty(batchTest, dtype=torch.long)
            y = y + 1
            for j in range(0,batchTest):
                if (j+i<len(testNu)):
                    pred = model(testNu[j+i])
                    res[j] = pred
                    label[j] = testLabelNu[j+i]
                else:
                     continue
            test_loss += loss_fn(res,label).item()
            correct += (res.argmax(1) == label).type(torch.float).sum().item()
           

        correct /= len(testNu)
        test_loss /=y 
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


#training data
with gzip.open('kddcup.data_10_percent.gz', 'rb') as f:
    data, dataLabels = PreProcessing(f)



Nu = np.array(data)
LabelNu = np.array(dataLabels)


X_train, X_test, y_train, y_test = train_test_split( Nu, LabelNu, test_size=0.20, random_state=42)


X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)


X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

    
# Initialize the loss function
learning_rate = 10**(-3)
epochs = 10

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
   
    train_loop(X_train,y_train , model, loss_fn, optimizer)
    test_loop(X_test, y_test, model, loss_fn)
print("Done!")
