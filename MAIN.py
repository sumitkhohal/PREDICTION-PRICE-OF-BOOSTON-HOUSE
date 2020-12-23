import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from IPython.display import HTML

boston=load_boston()
print(boston.DESCR)

features = pd.DataFrame(boston.data,columns=boston.feature_names)
features

features['AGE']

target = pd.DataFrame(boston.target,columns=['target'])
target

max(target['target'])

min(target['target'])

df=pd.concat([features,target],axis=1)
df

df.describe().round(decimals=2)

corr = df.corr('pearson')
corrs = [abs(corr[attr]['target']) for attr in list(features)]
l = list(zip(corrs, list(features)))
l.sort(key=lambda x : x[0], reverse=True)
corrs, labels = list(zip((*l)))
index = np.arange(len(labels))
index = np.arange(len(labels))
plt.figure(figsize=(15,5))
plt.bar(index, corrs, width=0.5)
plt.xlabel('aattributes')
plt.ylabel('Correlation with the target variable')
plt.xticks(index, labels)
plt.show()

X=df['LSTAT'].values
Y=df['target'].values

# BEFORE
print(Y[:5])

x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X.reshape(-1,1))
X = X[:, -1]
y_scaler=MinMaxScaler()
Y = y_scaler.fit_transform(Y.reshape(-1,1))
Y = Y[:, -1]

#after normalization
print(Y[:5])

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2)

def error(m, x, c, t):
    N = x.size
    e = sum(((m * x +c) - t) ** 2)
    return e * 1/(2 * N)
    
def update (m, x, c, t, learning_rate):
    grad_m = sum(2 * ((m * x + c) - t) * x)
    grad_c = sum(2 * ((m * x + c)-t))
    m = m - grad_m * learning_rate
    c = c - grad_c * learning_rate
    return m, c
    
def gradient_descent(init_m, init_c, x, t, learning_rate, iterations, error_threshold):
        m = init_m
        c = init_c
        error_values = list()
        mc_values = list()
        for i in range(iterations):
            e  = error(m, x, c, t)
            if e < error_threshold:
                print('Error less than the threshold. Stopping gradient descent')
                break
            error_values.append(e)
            m, c = update(m, x, c, t, learning_rate)
            mc_values.append((m,c))
        return m, c, error_values, mc_values

%%time
init_m=0.9
init_c=0
learning_rate=0.001
iterations=250
error_threshold=0.001

m,c,error_values,mc_values=gradient_descent(init_m,init_c,xtrain,ytrain,learning_rate,iterations,error_threshold)

plt.scatter(xtrain, ytrain,color='b')
plt.plot(xtrain,(m * xtrain + c), color= 'r')

plt.plot(np.arange(len(error_values)),error_values)
plt.ylabel('error')
plt.xlabel('iterations')

mc_values_anim = mc_values[0:250:5]

fig, ax = plt.subplots()
ln, = plt.plot([], [], 'ro-', animated=True)

def init():
    plt.scatter(xtest,ytest,color='g')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    return ln,

def update_frame(frame):
    m , c = mc_values_anim[frame]
    x1, y1 = -0.5, m * -.5 + c
    x2, y2 = 1.5, m * 1.5 + c
    ln.set_data([x1,x2], [y1,y2])
    return ln,

anim = FuncAnimation(fig, update_frame, frames=range(len(mc_values_anim)),init_func=init, blit=True)
HTML(anim.to_html5_video())

predicted=(m*xtest)+c

mean_squared_error(ytest, predicted)

p = pd.DataFrame(list(zip(xtest, ytest, predicted)),columns=['x', 'target_y', 'predicted_y'])
p.head()

plt.scatter(xtest,ytest,color='b')
plt.plot(xtest, predicted, color='r')

predicted = predicted.reshape(-1,1)
xtest = xtest.reshape(-1,1)
ytest = ytest.reshape(-1,1)

xtest_scaled = x_scaler.inverse_transform(xtest)
ytest_scaled = y_scaler.inverse_transform(ytest)
predicted_scaled = y_scaler.inverse_transform(predicted)

xtest_scaled = xtest_scaled[:,-1]
ytest_scaled = ytest_scaled[:,-1]
predicted_scaled = predicted_scaled[:,-1]

p = pd.DataFrame(list(zip(xtest_scaled, ytest_scaled, predicted_scaled)), columns=['x', 'target', 'predicted'])
p = p.round(decimals=2)
p.head()
