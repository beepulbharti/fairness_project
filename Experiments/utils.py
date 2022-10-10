# Necessary functions to run the experiments

# Import necessary packages
import numpy as np
from updated_balancers import BinaryBalancer
from tqdm import tqdm
import os
import pandas as pd

# pytorch
import torch
from torch.optim import Adam
from torch import nn

# BERT from Huggingface
from transformers import BertTokenizer
from transformers import BertModel


# Sigmoid function
def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig

# New function to generate data (X,y)
def generate_data(n,e1,e2,b,group,exp):

    # Mean and variance for features x1,x2,x3
    mu = np.array([1,-1,0])
    var = np.array([[1,0.2,0],[0.2,1,0],[0,0,1]])
    X = np.random.multivariate_normal(mu,var,n)

    # Function from x3 to A
    a = ((X[:,2] + b) >= 0).astype('float')

    # Function from x1 and x2 to A
    eps_1 = np.random.normal(0,e1,n)
    eps_2 = np.random.normal(0,e2,n)

    # add noise to a = 0 or a = 1
    noise_a = eps_2*(a==group)

    # Generate y depending on experiment
    if exp == 1:
        y = (sigmoid(X[:,0] + X[:,1] + eps_1 + noise_a) >= 0.5).astype('float')
    else:
        y = (sigmoid(X[:,0] + X[:,1] + X[:,2] + eps_1 + noise_a) >= 0.5).astype('float')
    
    return (X, a, y)

# Function to generate y_prob using random coefficients
def generate_y_hat(X,coeffs,exp):
    if exp == 1:
        y_prob = sigmoid(np.dot(X[:,:2],coeffs[:2]))
    else:
        y_prob = sigmoid(np.dot(X,coeffs))
    y_hat = (y_prob >= 0.5).astype('float')

    return (y_prob, y_hat)

def calc_base_rates(y,a):
    # Organizing data
    data = np.column_stack((y,a))

    # converting to dataframes
    df = pd.DataFrame(data, columns = ['y','a'])

    # calculating base rates
    r = df[(df['a'] == 1) & (df['y'] == 1)].shape[0]/df.shape[0]
    s = df[(df['a'] == 0) & (df['y'] == 1)].shape[0]/df.shape[0]
    v = df[(df['a'] == 1) & (df['y'] == 0)].shape[0]/df.shape[0]
    w = df[(df['a'] == 0) & (df['y'] == 0)].shape[0]/df.shape[0]
    
    return (r,s,v,w)

# Functions to calculate the TPRs and FPRs with respect to A
def calculate_bias_metrics(balancer):
    alpha = balancer.group_rates[1.0].tpr
    beta = balancer.group_rates[0.0].tpr
    tau = balancer.group_rates[1.0].fpr
    phi = balancer.group_rates[0.0].fpr
    return alpha,beta,tau,phi

# Generate a_hat by making independent errors with probability = p
def generate_a_hat_indep_p(a,p):
    a_mask = np.copy(a)
    a_mask[a_mask==0] = -1
    vals = np.random.random(a_mask.shape[0])
    vals = (vals >= p).astype('float')
    vals[vals == 0] = -1
    a_hat = a_mask*vals
    a_hat[a_hat == -1] = 0 

    return a_hat

# Generate a_hat for experiment 2
def generate_a_hat(x3, b, e):
    a_hat = ((x3 + e + b) >= 0).astype('float')
    return a_hat

# Calculating upper and lower bounds when assumption holds
def calc_assump_bounds(r,s,U,delta):
    if np.abs(r-s) <= delta:
        k = s/(s-U)
        return k
    else:
        z_rs = (2*r*s**2 - 2*s**2*U - (s**2*(r+s)**2*(r-U)*(s-U))**(1/2))/(2*s**2*(r-s))
        ep_1 = (1-U/r)/((s/r + U/r)*(r/s-U/s))
        ep_2 = (1-U/s)/((s/r - U/r)*(r/s + U/s))
        if (z_rs >= 0 and z_rs <= U/s):
            k = 1/(1-z_rs - (U-s*z_rs)/r)/(((s/r)*(1-z_rs)+(U-s*z_rs)/r)*((r/s)*(1-(U-s*z_rs)/r)+z_rs))
        else:
            k = 1/min(ep_1,ep_2)
        return k

# Calculating general upper and lower bounds
def calc_gen_bounds(alpha,beta,U,r,s):
    if s*alpha + r*beta > 0.5*(s+r):
        ub = alpha - beta + U*(alpha/r + beta/s)
        lb = alpha - beta - U*(alpha/r + beta/s)
    else:
        ub = alpha - beta - U*(alpha/r + beta/s) + U*(r+s)/(r*s)
        lb = alpha - beta + U*(alpha/r + beta/s) - U*(r+s)/(r*s)
    return ub, lb

# Function to post process y_hat
def eo_postprocess(y,y_,a):
    fair_model = BinaryBalancer(y=y,y_=y_,a=a)
    fair_model.adjust(goal='odds', summary=False)
    fair_yh = fair_model.predict(y_,a)
    return fair_yh, fair_model

# Function to generate data
def old_generate_data(n,e1,e2,b,group,exp):

    # Mean and variance for features x1,x2,x3
    # x3 is independent from x1 and x2
    mu = np.array([1,-1,0])
    var = np.array([[1,0.2,0],[0.2,1,0],[0,0,1]])
    X = np.random.multivariate_normal(mu,var,n)

    # Function from x3 to A
    a = ((X[:,2] + b) >= 0).astype('float')

    # Function from x1 and x2 to A
    eps_1 = np.random.normal(0,e1,n)
    eps_2 = np.random.normal(0,e2,n)

    # add noise to a = 0 or a = 1
    noise_a = eps_2*(a==group)
    
    if exp == 1:
        y = (sigmoid(X[:,0] + X[:,1] + eps_1 + noise_a) >= 0.5).astype('float')
    else:
        y = (sigmoid(X[:,0] + X[:,1] + X[:,2] + eps_1 + noise_a) >= 0.5).astype('float')

    # y_hat
    if exp == 1:
        y_prob = sigmoid(X[:,0] + x2)
    else:
        y_prob = sigmoid(x1 + x2 + x3)
    
    y_hat = (y_prob >= 0.5).astype('float')

    return X, a, y_prob, y_hat, y

# Functions to calculate the TPRs and FPRs with respect to A
def old_calculate_bias_metrics(df):
    a = df.a.values
    y = np.array(df.y.values)
    y_ = np.array(df.y_hat.values)
    pb = BinaryBalancer(y=y,y_=y_,a=a)
    alpha = pb.group_rates[1.0].tpr
    beta = pb.group_rates[0.0].tpr
    tau = pb.group_rates[1.0].fpr
    phi = pb.group_rates[0.0].fpr
    return alpha,beta,tau,phi


# Generate a_hat for experiment 2
def old_generate_a_hat(x3, b, mu ,noise):
    e = np.random.normal(mu,noise)
    a_hat = ((x3 + e + b) >= 0).astype('float')
    return a_hat


# Function to post process y_hat
def old_eo_postprocess(df):
    a = df.a.values
    y = np.array(df.y.values)
    y_ = np.array(df.y_prob.values)
    fair_model = BinaryBalancer(y=y,y_=y_,a=a)
    fair_model.adjust(goal='odds', summary=False)
    fair_yh = fair_model.predict(y_,a)
    return fair_yh, fair_model

# Dataset class for BERT
class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [label for label in df['a']]
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 5, truncation=True,
                                return_tensors="pt") for text in df['long_name']]
        self.remain_data = [df[['age','overall','y','group']].iloc[idx] for idx in range(df.shape[0])]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        batch_texts = self.texts[idx]
        batch_y = torch.tensor(self.labels[idx])
        batch_rest = torch.tensor(self.remain_data[idx])

        return batch_texts, batch_y, batch_rest

# Class for classifier
class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.sigmoid(linear_output)

        return final_layer


def train(model, train_data, val_data, learning_rate, epochs):
    
    batch_sz = 2

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_sz, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_sz)

    os.environ['CUDE_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)


    for epoch_num in range(epochs):

        model.train()
        total_loss_train = 0
        total_tp_train = 0

        for train_input, train_label, _ in tqdm(train_dataloader):

            train_label = train_label.to(device).float()
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask).reshape(1,-1)[0] 
            batch_loss = criterion(output, train_label)
            batch_tp = torch.sum((output >= 0.5) == train_label)
            
            total_tp_train += batch_tp.item()
            total_loss_train += batch_loss.item()

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
                
        total_loss_val = 0
        total_tp_val = 0

        with torch.no_grad():

            model.eval()

            for val_input, val_label, _ in val_dataloader:

                val_label = val_label.to(device).float()
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask).reshape(1,-1)[0]
                batch_loss = criterion(output, val_label)
                batch_tp = torch.sum((output >= 0.5) == val_label)
                    
                total_tp_val += batch_tp.item()    
                total_loss_val += batch_loss.item()
                    
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / (len(train_data)/batch_sz): .3f} \
            | Train Accuracy: {total_tp_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / (len(val_data)/batch_sz): .3f} \
            | Val Accuracy: {total_tp_val / len(val_data): .3f}')