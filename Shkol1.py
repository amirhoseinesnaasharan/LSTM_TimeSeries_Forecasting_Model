import numpy as np
import pandas as pd
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_excel('C:\\Users\\Amir\\Desktop\\python\\code python\\Shkol1.xls')

# Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split data into training and testing sets
train_size = int(len(scaled_data) * 0.7)
test_size = len(scaled_data) - train_size
train, test = scaled_data[0:train_size], scaled_data[train_size:len(scaled_data)]

#convert an array value s into data matrix
def creat_dataset (scaled_data,look_back=5):
    dataX, dataY = [] ,[]
    for i in range (len(scaled_data)-look_back-1):
        a=scaled_data[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(scaled_data[i + look_back, 0])
    return np.array (dataX) , np.array(dataY)

#Reshape into X=t and Y=t+1
look_back =10
trainX , trainY = creat_dataset(train, look_back)
testX , testY =creat_dataset (test, look_back)

# Reshape input to be [sample, time stepe, feature]
trainXr = np.reshape (trainX, (trainX.shape[0],trainX.shape[1],1))
testXr= np.reshape(testX,(testX.shape[0],testX.shape[1],1))

trainX=trainXr
testX=testXr

#creat and fit the lstm
model = Sequential()
model.add(LSTM(5, input_shape=(look_back, 1)))
model.add(Dense(1))
# Compile and train model
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=2)

# Make predictions
trainPredict= model.predict(trainX)
testPredict=model.predict(testX)

#invert predict
trainPredict= scaler.inverse_transform(trainPredict)
trainY= scaler.inverse_transform([trainY])
testPredict=scaler.inverse_transform(testPredict)
testY= scaler.inverse_transform([testY])

#calculate root mean squared error
trainScore=math.sqrt(mean_squared_error(trainY[0],trainPredict[:,0]))
print("train score: %2f RMSE" % (trainScore))
testScore=math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print("test score: %2f RMSE" % (testScore))

# Predict the next 20 steps
X_future = np.zeros((1, look_back, 1))
for i in range(look_back):
    X_future[0, i, 0] = scaled_data[-i-1]

future_predictions = []
for i in range(20):
    yhat = model.predict(X_future)
    future_predictions.append(yhat[0,0])
    X_future = np.roll(X_future, -1)
    X_future[-1, 0, 0] = yhat[0,0]

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))
print("Future predictions: ", future_predictions)