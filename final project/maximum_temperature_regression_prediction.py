#I use the existing model in the code. for example,
# Optimizer =tf.keras.optimizers.SGD(0.01), Loss ='mean_squared_error ')
# in model.pile (Optimizer =tf.keras.optimizers) It's an existing model.
# But we did almost all of the code ourselves, because the model fits with the data we've been working with ourselves.
# This means that all code is written around our data. So, almost all of the code was done by ourselves,
# and only the model was applied by picking existing models.

#回归预测温度
# Arithmetic and logical operations on arrays
import numpy as np
# Provide data structure and data analysis tool
import pandas as pd
# Modify the graphics
import matplotlib.pyplot as plt
# Use data flow graphs for numerical calculations
import tensorflow as tf
# modeling by keras
from tensorflow.keras import layers
# error reminder
import warnings

# ignore the error reminder on the model name when calling the model
warnings.filterwarnings('ignore')

# ----- 1. data acquisition -----
filepath = '/Users/yinning/Desktop/temperature.csv'
# data comes from https://data.giss.nasa.gov/gistemp/
features = pd.read_csv(filepath)

# ----- 2. data preprocessing -----
# （1）processing time data, combining years, months and days
# show date
import datetime

# get data of "year", "month", and "day"
years = features['Year']
months = features['Month']
# combine the year, month and day as strings
dates = []
for year, month in zip(years, months):
    date = str(year) + '-' + str(month)
    dates.append(date)
# (2) convert to datetime format
times = []
for date in dates:
    time = datetime.datetime.strptime(date, '%Y-%m')
    times.append(time)

# ----- 3. data visualization, mapping each feature -----
# specify drawing style
plt.style.use('fivethirtyeight')
# set the canvas, 2 rows and 2 columns of the drawing window, the first row draw ax1 and ax2
fig, ((fig1, fig2), (fig3, fig4)) = plt.subplots(2, 2, figsize=(20, 10))  # foot

# (1) actual temperature
fig1.plot(times, features['actual'])
# set the label of X axis, Y axis and title
fig1.set_xlabel('Date')
fig1.set_ylabel('Temperature')
fig1.set_title('actual temperature')
# (2) temperature of the yesterday
fig2.plot(times, features['temp_1'])
fig2.set_xlabel('Date')
fig2.set_ylabel('Temperature')
fig2.set_title('temp_1')
# (3) temperature of the day before yesterday
fig3.plot(times, features['temp_2'])
fig3.set_xlabel('Date')
fig3.set_ylabel('Temperature')
fig3.set_title('temp_2')
# (4) temp_3
fig4.plot(times, features['temp_3'])
fig4.set_xlabel('Date')
fig4.set_ylabel('Temperature')
fig4.set_title('temp_3')
# lightweight layout adjusts drawing to prevent chart overlap
plt.tight_layout(pad = 2)


# ----- 4. divide eigenvalues and target values -----
# get the target value y and translate it from list to array
targets = np.array(features['actual'])
# get the feature value x, that is, remove the target value column from the original data
# delete the row by default, and specify axis=1 to point to the column
features = features.drop('actual', axis = 1)
# change features from date frame to array
features = np.array(features)

# ----- 5. standardized treatment -----
# data preprocessing
from sklearn import preprocessing

# preprocessing.StandardScaler() standardize the data by subtracting the mean and then dividing by the variance
# fit_transform first fit part of the data, find the overall index of the part, and then transform the train data
# realize the standardization and normalization of data
input_features = preprocessing.StandardScaler().fit_transform(features)

# ----- 6. build the network model with keras -----
# (1) build layers
# choose style of model
model = tf.keras.Sequential()
# 隐含层1设置16层，权重初始化方法设置为随机高斯分布
# 加入正则化惩罚项
# model.add(layers.Dense(16, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# model.add(layers.Dense(32, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(layers.Dense(1, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# (2) specify optimizer
# tf.keras.optimizers.SGD(0.01) rate of learning
# loss='mean_squared_error': mean square error loss function
model.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss='mean_squared_error')
# (3) network training
# validation_split=0.25 divide 75% of the test set into the training set
# epochs=100 the number of iterations is 100
# batch_size=128 each batch size is 128
model.fit(input_features, targets, validation_split = 0.25, epochs = 100, batch_size = 128)
# (4) network model structure
# output the parameters of each layer of the model
model.summary()
# (5) prediction model results
predict = model.predict(input_features)
# ----- 7. show the predicted results -----
# (1) blue represents actual value
fig = plt.figure(figsize=(15, 7))
# divide figure into 1 part
axes = fig.add_subplot(111)
axes.plot(dates, targets, 'bo', label='actual')
# (2) red represents predict value
axes.plot(dates, predict, 'ro', label='predict')
# sets the scale position of X axis.
# there are 12 months in a year, so, divided by 12
axes.set_xticks(dates[::12])
# set the label for X axis using the string label list
axes.set_xticklabels(dates[::12], rotation=30)

# after plt.plot() is defined, plt.legend() displays the contents of the label
plt.legend()
# show figure
plt.show()