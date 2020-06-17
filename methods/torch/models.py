import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Class containing vanilla convolutional network for pathology reports
"""

class CnnClassifier(nn.Module):
    
    def __init__(self,args):
        super(CnnClassifier, self).__init__()
        
        self.filters = args['filters']
        self.filterNum = args['filterNum']
        self.maxDocLength = args['maxDocLength']
        self.wordDim = args['wordDim']
        self.classSize = args['classSize']
        self.embeddingDim = args['embeddingDim']
        self.dropOut = nn.Dropout(p = args['dropOut'])
        self.inChannel = 1
        self.convs = {}
        self.embedding = nn.Embedding(self.wordDim, self.embeddingDim)
        
        for i in range(len(self.filters)):
            conv = nn.Conv1d(self.inChannel, self.filterNum[i], 
                             self.embeddingDim * self.filters[i], 
                                stride=self.embeddingDim)
            self.convs[i] = conv
            self.add_module("conv" + str(i) ,self.convs[i])
        self.fc = nn.Linear(sum(self.filterNum), self.classSize)
        
    def getConv(self, i):
        return self.convs[i]
        
    def forward(self, x):
        x = self.embedding(x).view(-1,1,self.embeddingDim*self.maxDocLength)
        convResults = [ F.max_pool1d(F.relu(self.getConv(i)(x)), 
                        self.maxDocLength - self.filters[i] + 1).view(-1, self.filterNum[i])
                        for i in range(len(self.filters))]
        x = torch.cat(convResults, 1)
        x = self.dropOut(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x