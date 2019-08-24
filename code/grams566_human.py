from __future__ import division

import os, sys
import numpy as np
from numpy import array
from numpy import argmax, random
import pandas as pd
import random
import csv

import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import time

os.chdir("Rayin-saber/Time-series-vaccine-prediction/data")

#read the file name
#data_name = ['2000.csv','2001.csv','2002.csv','2003.csv','2004.csv','2005.csv','2006.csv','2007.csv','2008.csv','2009.csv','2010.csv','2011.csv','2012.csv','2013.csv','2014.csv','2015.csv']
data_name = ['2012.csv','2013.csv','2014.csv','2015.csv', '2016.csv', '2017.csv']

#open the ProtVec and mapping embeddings
with open('protVec_100d_3grams.csv', mode='r') as f:
    reader = csv.DictReader(f,delimiter = '\t')
    name_100d3 = list(reader.fieldnames)
    name_100d3 = name_100d3[1:101]
    
gram_100d3 = pd.read_csv('protVec_100d_3grams.csv',delimiter = '\t')
seq_name = list(gram_100d3['words'])
seq_value = gram_100d3[name_100d3].values
print ("gram shape",seq_value.shape)

year_num = 5
#sample_num = 15
random_num = 1000


data_raw_uncertain = []
for filename in data_name:
    rawdata = pd.read_csv('csv/host/H3N2/'+filename) #series
    rawdata = rawdata['seq'] # series
    data_raw_uncertain.append(rawdata)


l_max=566
Btworandom = 'DN'
Jtworandom = 'IL'
Ztworandom = 'EQ'
Xallrandom = 'ACDEFGHIKLMNPQRSTVWY'

data_raw = data_raw_uncertain

# pad to same length with -
for i in range(year_num):
    for j in range(len(data_raw_uncertain[i])):

        lis = data_raw_uncertain[i].iloc[j]
        lis = lis.replace('B',random.choice(Btworandom))
        lis = lis.replace('J',random.choice(Jtworandom))
        lis = lis.replace('Z',random.choice(Ztworandom))
        lis = lis.replace('X',random.choice(Xallrandom))
        data_raw[i].iloc[j] = lis
        if len(data_raw_uncertain[i].iloc[j]) < l_max:
            data_raw[i].iloc[j] += '-'*(l_max-len(data_raw_uncertain[i].iloc[j]))
#print (data_raw[0].iloc[0])

#prepare training data
def prepare_traindata(seq,random_num):
    result = []
    for i in range(random_num):
        for j in range (l_max-2):
            aminoacid_years = ''
            for k in range(year_num):
                a = random.randint(0,len(data_raw_uncertain[k])-1)
               

                aminoacid_3 = seq[k].iloc[a][j]+seq[k].iloc[a][j+1]+seq[k].iloc[a][j+2]
                aminoacid_years += aminoacid_3
            result.append(aminoacid_years)
    return result

train_data_seq = prepare_traindata(data_raw,random_num)

aminoacid_num = len(train_data_seq)

#convert to label
def intlabel(seq):
    data_integer = np.zeros(((l_max-2)*random_num,year_num))
    for i in range((l_max-2)*random_num):
        for j in range(year_num):
            temp = seq[i][j*3:j*3+3]
            if temp[0] != '-' and temp[1] != '-' and temp[2] != '-':
                data_integer[i][j] = seq_name.index(temp)
            else:
                data_integer[i][j] = 9048

    return np.array(data_integer)

label_int = intlabel(train_data_seq) #any seq contain '-' is defined as label 9048, seq in dictionary get label=position in dictionary+1
label_trans = label_int.transpose()
train_label = torch.LongTensor(label_trans)
#print (train_label.size())

train_data0 = np.zeros((train_label.size(0),train_label.size(1),100))
for i in range(train_label.size(0)):
    for j in range(train_label.size(1)):
        if train_label[i][j] == 9048:
            train_data0[i][j][:] = array([0]*100)
        else:
            train_data0[i][j][:] = seq_value[train_label[i][j]]
        
train_data = torch.tensor(train_data0)
train_data = train_data.type(torch.FloatTensor)
print ("train data shape",train_data.shape)

#parameter settings        
bs=256
input_size = 100
output_size = 9049
seq_len = year_num-1
hidden_size = 400
layer_num = 1
EPOCH = 50 # train the training data n times

#build the three-layer rnn model
class three_layer_recurrent_net(nn.Module):

    def __init__(self, hidden_size):
        super(three_layer_recurrent_net, self).__init__()
        
        self.layer1 = nn.LSTM(      input_size , hidden_size , layer_num )
        self.layer2 = nn.Linear(    hidden_size , output_size   )

        
    def forward(self, word_seq, h_init, c_init ):
        
        
        h_seq , (h_final,c_final)  =   self.layer1( word_seq , (h_init,c_init) )      
        score_seq                  =   self.layer2( h_seq[-1,:,:] )
        
        return score_seq

net = three_layer_recurrent_net( hidden_size )

print(net)

net.layer2.weight.data.uniform_(-0.1, 0.1)

print('')
criterion = nn.CrossEntropyLoss()

my_lr = 0.1

start=time.time()

#training process
for epoch in range(EPOCH):
    
    # divide the learning rate by 3 except after the first epoch
    #if epoch >= 20:
     #   my_lr = my_lr / 2
    
    # create a new optimizer at the beginning of each epoch: give the current learning rate.   
    optimizer=torch.optim.SGD( net.parameters() , lr=my_lr )
        
    # set the running quatities to zero at the beginning of the epoch
    running_loss=0
    num_batches=0    
       
    # set the initial h and c to be the zero vector
    h = torch.zeros(layer_num, bs, hidden_size)
    c = torch.zeros(layer_num, bs, hidden_size)

    # send them to the gpu    
    #h=h.to(device)
    #c=c.to(device)
    
    for count in range( 0 , (l_max-2)*random_num-bs ,  bs):
        
        # Set the gradients to zeros
        optimizer.zero_grad()
        
        # create a minibatch
        minibatch_data =  train_data[ : seq_len,count:count+bs,:  ]
        minibatch_label = train_label[-1,count:count+bs ]        
        
        # send them to the gpu
        #minibatch_data=minibatch_data.to(device)
        #minibatch_label=minibatch_label.to(device)
        
        # Detach to prevent from backpropagating all the way to the beginning
        # Then tell Pytorch to start tracking all operations that will be done on h and c
        h=h.detach()
        c=c.detach()
        h=h.requires_grad_()
        c=c.requires_grad_()
                       
        # forward the minibatch through the net        
        scores  = net( minibatch_data, h , c)
        
        # reshape the scores and labels to huge batch of size bs*seq_length
        scores  = scores.view(bs , output_size)  
            
        
        # Compute the average of the losses of the data points in this huge batch
        loss = criterion(scores ,  minibatch_label )
        
        # backward pass to compute dL/dR, dL/dV and dL/dW
        loss.backward()

        # do one step of stochastic gradient descent: R=R-lr(dL/dR), V=V-lr(dL/dV), ...
        #utils.normalize_gradient(net)
        optimizer.step()
        
            
        # update the running loss  
        running_loss += loss.item()
        num_batches += 1
        
        
        
    # compute stats for the full training set
    total_loss = running_loss/num_batches
    elapsed = time.time()-start
    
    print('')
    print('epoch=',epoch, '\t time=', elapsed,'\t lr=', my_lr, '\t exp(loss)=',  math.exp(total_loss))


#mapping sequence to array to extract the amino acid
def seq2array(seq):
    data_integer = np.zeros(l_max-2)
    sample_data = np.zeros((l_max-2,100))
    for i in range(l_max-2):        
        temp = seq[i:i+3]
        if temp[0] != '-' and temp[1] != '-' and temp[2] != '-':
            data_integer[i] = seq_name.index(temp)   
            sample_data[i][:] = seq_value[int(data_integer[i])]
        else:
            data_integer[i] = 9048
            sample_data[i][:] = array([0]*100)
    return np.array(sample_data)

#random.seed(10)

samp0 = seq2array(data_raw[0].iloc[np.random.randint(len(data_raw_uncertain[0]))])
samp1 = seq2array(data_raw[1].iloc[np.random.randint(len(data_raw_uncertain[1]))])
samp2 = seq2array(data_raw[2].iloc[np.random.randint(len(data_raw_uncertain[2]))])
samp3 = seq2array(data_raw[3].iloc[np.random.randint(len(data_raw_uncertain[3]))])
samp4 = seq2array(data_raw[4].iloc[np.random.randint(len(data_raw_uncertain[4]))])
#samp5 = seq2array(data_raw[5].iloc[input_index[5]])
#samp6 = seq2array(data_raw[6].iloc[input_index[6]])
#samp7 = seq2array(data_raw[7].iloc[input_index[7]])
#samp8 = seq2array(data_raw[8].iloc[input_index[8]])
#samp9 = seq2array(data_raw[9].iloc[input_index[9]])

#minibatch_data = torch.LongTensor([[1,15,36],[1,15,36],[2,16,37],[3,17,38],[2,16,37],[3,17,38],[2,16,37],[2,16,37],[4,18,39]])
minibatch_data = torch.Tensor([samp0,samp1,samp2,samp3,samp4])

#minibatch_data = torch.Tensor([samp0,samp1,samp2,samp3,samp4,samp5,samp6,samp7,samp8,samp9])
#minibatch_data = torch.LongTensor([[36],[36],[37],[38],[37],[38],[37],[37],[39]])
h = torch.zeros(layer_num, l_max-2, hidden_size)
c = torch.zeros(layer_num, l_max-2, hidden_size)

scores  = net(minibatch_data , h, c)
# print (scores)

#prediction
def show_next(scores):
    num_word_display=1
    prob=F.softmax(scores,dim=1)
    #p=prob[-1].squeeze()
    p,word_idx = torch.topk(prob,num_word_display)
    #return prob, p, word_idx

    aminoacid_pre = [] 
    for j in range(l_max-2):
        #percentage= p[j]*100
        idx = word_idx[j]
        name = seq_name[idx]
        aminoacid_pre.append(name)
            
    #print (aminoacid_pre)
    return p,aminoacid_pre


def most_common(lst):
    return max(set(lst), key=lst.count)

def ami3_to_seq(p,pre_result):
    
    # find effective value
    for index, x in enumerate(p):
        if x == 1:
            print ('prediction sequence lenth =',index+2)
            break
        
    effect_len = index    
    # construct seq
    final_seq = pre_result[0][0]
    # judge first 2 alphabet   
    if pre_result[0][1] == pre_result[1][0]:
        final_seq += pre_result[0][1]
    else:      
        position = torch.argmax(p[0:2])
        final_seq += pre_result[position][1-position]
            
    for i in range(effect_len-1):
        temp = [pre_result[i][2],pre_result[i+1][1],pre_result[i+2][0]] 
        most_alpha = most_common(temp)
        if temp.count(most_alpha) > 1:
            final_seq += most_alpha
        else:
            position = torch.argmax(p[i:i+3])
            final_seq += pre_result[i+position][2-position]
    # judge last 2 alphabet       
    if pre_result[index-1][2] == pre_result[index][1]:
        final_seq += pre_result[index-1][2]
    else:
        position = torch.argmax(p[effect_len-2:effect_len])
        final_seq += pre_result[effect_len-2+position][2-position]
        
    final_seq += pre_result[index][2]
    
    return final_seq

p,pre_result = show_next(scores)
pre_seq = ami3_to_seq(p,pre_result)
print('predicted sequence',pre_seq)

full_length = 566
ave_num_identical_aa = 0
ave_identity_rate = 0
HA_H3N2_vaccine_2017 = pd.read_csv('vaccine_recommend/H3N2_strains/16-17/H3N2_vaccine_2017.csv', names=['accession', 'seq', 'description'], na_filter = False)
for i in range(HA_H3N2_vaccine_2017.shape[0]):
    num_identical_aa = sum(1 if c1 == c2 else 0 for c1, c2 in zip(pre_seq, HA_H3N2_vaccine_2017['seq'].iloc[i]))
    ave_num_identical_aa += num_identical_aa    
    identity_rate = float(num_identical_aa/full_length)
    ave_identity_rate += identity_rate
    print('predict_sequence vs' + HA_H3N2_vaccine_2017['description'].iloc[i])
    print('same alpha', num_identical_aa)
    print('identity rate', identity_rate)
    
ave_num_identical_aa = ave_num_identical_aa/HA_H3N2_vaccine_2017.shape[0]
ave_identity_rate = ave_identity_rate/HA_H3N2_vaccine_2017.shape[0]
print('ave_num:', ave_num_identical_aa)
print('ave_rate:', ave_identity_rate)

def prediction_seq(pred, year, sampnum):
    file_path = str('prediction_seq') + '_' + str(year) + '_' + 'sample' + str(sampnum) + '.csv'
    with open(file_path, 'w') as output:
        output.write(pred)
    output.close()

prediction_seq(pre_seq, 2017, random_num)





##predicting multiple years
#data_name1 = ['2016.csv','2017.csv']
#data_raw1 = []
#for filename in data_name1:
#    rawdata1 = pd.read_csv('csv/region/North_America/' + filename,names=['seq']) #series
#    rawdata1 = rawdata1['seq'] # series
#    data_temp1 = [] 
#    #idx = random.sample(range(1, len(rawdata)), sample_num) # 10 samples per year
#    sample1 = rawdata.iloc[0] # string
#    data_raw1.append(sample1)
#    
#
#if len(data_raw1[0]) < l_max:
#            data_raw1[0] += '-'*(l_max-len(data_raw1[0]))
#if len(data_raw1[1]) < l_max:
#            data_raw1[1] += '-'*(l_max-len(data_raw1[1]))
#
#
## predict next 2nd year
#
#
#pre_seq = pre_seq.replace('B',random.choice(Btworandom))
#pre_seq = pre_seq.replace('J',random.choice(Jtworandom))
#pre_seq = pre_seq.replace('Z',random.choice(Ztworandom))
#pre_seq = pre_seq.replace('X',random.choice(Xallrandom))
#
#
#if len(pre_seq) < l_max:
#            pre_seq += '-'*(l_max-len(pre_seq))
#
#samp10 = seq2array(pre_seq)
#minibatch_data1 = torch.Tensor([samp0,samp1,samp2,samp3,samp4,samp5,samp6,samp7,samp8,samp9,samp10])
#h1 = torch.zeros(layer_num, l_max-2, hidden_size)
#c1 = torch.zeros(layer_num, l_max-2, hidden_size)
#
#scores1  = net(minibatch_data1 , h1, c1)
#p1,pre_result1 = show_next(scores1)
#pre_seq1 = ami3_to_seq(p1,pre_result1)
#print ('predict next 2nd sequence',pre_seq1)
#print ('compare sequence',data_raw1[0])
#compare21 = sum(1 if c1 == c2 else 0 for c1, c2 in zip(pre_seq1, data_raw1[0]))
#print ('same alpha',compare21)
#
#
## predict next 3rd year
#pre_seq1 = pre_seq1.replace('B',random.choice(Btworandom))
#pre_seq1 = pre_seq1.replace('J',random.choice(Jtworandom))
#pre_seq1 = pre_seq1.replace('Z',random.choice(Ztworandom))
#pre_seq1 = pre_seq1.replace('X',random.choice(Xallrandom))
#
#if len(pre_seq1) < l_max:
#            pre_seq1 += '-'*(l_max-len(pre_seq1))
#
#samp11 = seq2array(pre_seq1)
#minibatch_data2 = torch.Tensor([samp0,samp1,samp2,samp3,samp4,samp5,samp6,samp7,samp8,samp9,samp10,samp11])
#h2 = torch.zeros(layer_num, l_max-2, hidden_size)
#c2 = torch.zeros(layer_num, l_max-2, hidden_size)
#
#scores2  = net(minibatch_data2 , h2, c2)
#p2,pre_result2 = show_next(scores2)
#pre_seq2 = ami3_to_seq(p2,pre_result2)
#print ('predict next 3rd sequence',pre_seq2)
#print ('compare sequence',data_raw1[1])
#compare22 = sum(1 if c1 == c2 else 0 for c1, c2 in zip(pre_seq2, data_raw1[1]))
#print ('same alpha',compare22)
