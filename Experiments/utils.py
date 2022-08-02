# Necessary functions to run the experiments

# Import necessary packages
import numpy as np
from balancers import BinaryBalancer


# Sigmoid function
def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig

# Function to generate data
def generate_data(n,e1,e2,b,group,exp):

    # Mean and variance for features x1,x2,x3
    # x3 is independent from x1 and x2
    mu = np.array([1,-1,0])
    var = np.array([[1,0.2,0],[0.2,1,0],[0,0,1]])
    X = np.random.multivariate_normal(mu,var,n)
    x1 = X[:,0]
    x2 = X[:,1]
    x3 = X[:,2]

    # Function from x3 to A
    a = ((x3 + b) >= 0).astype('float')

    # Function from x1 and x2 to A
    eps_1 = np.random.normal(0,e1,n)
    eps_2 = np.random.normal(0,e2,n)

    # add noise to a = 0 or a = 1
    noise_a = eps_2*(a==group)
    
    if exp == 1:
        y = (sigmoid(x1 + x2 + eps_1 + noise_a) >= 0.5).astype('float')
    else:
        y = (sigmoid(x1 + x2 + x3 + eps_1 + noise_a) >= 0.5).astype('float')

    # y_hat
    if exp == 1:
        y_prob = sigmoid(x1 + x2)
    else:
        y_prob = sigmoid(x1 + x2 + x3)
    
    y_hat = (y_prob >= 0.5).astype('float')

    return X, a, y_prob, y_hat, y

# Functions to calculate the TPRs and FPRs with respect to A
def calculate_bias_metrics(df):
    a = df.a.values
    y = np.array(df.y.values)
    y_ = np.array(df.y_hat.values)
    pb = BinaryBalancer(y=y,y_=y_,a=a,summary=False)
    alpha = pb.group_rates[1.0].tpr
    beta = pb.group_rates[0.0].tpr
    tau = pb.group_rates[1.0].fpr
    phi = pb.group_rates[0.0].fpr
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
def generate_a_hat(x3, b, mu ,noise):
    e = np.random.normal(mu,noise)
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

def eo_postprocess(df):
    a = df.a.values
    y = np.array(df.y.values)
    y_ = np.array(df.y_prob.values)
    fair_model = BinaryBalancer(y=y,y_=y_,a=a,summary=False)
    fair_model.adjust(goal='odds', summary=False)
    fair_yh = fair_model.predict(y_,a)
    return fair_yh, fair_model