import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout

"""
We are going to use numpy for scientific operations, pandas to modify our dataset, matplotlib to visualize the results,
 sklearn to scale our data, and keras to work as a wrapper on low-level libraries like 
 TensorFlow or Theano high-level neural networks library.
"""

df = pd.read_csv('TSLA.csv')
df.shape  # The result will be (2392, 7).

"""
First of all, if you take a look at the dataset, you need to know that the “open” column represents the opening price
 for the stock at that “date” column, and the “close” column is the closing price on that day. 
 The “High” column represents the highest price reached that day, and the “Low” column represents the lowest price.
"""

df = df['Open'].values
df = df.reshape(-1, 1)

"""
The reshape allows you to add dimensions or change the number of elements in each dimension. 
We are using reshape(-1, 1) because we have just one dimension in our array, so numby will create the same number of
 our rows and add one more axis: 1 to be the second dimension.
"""

# Now let’s split the data into training and testing sets:

dataset_train = np.array(df[:int(df.shape[0] * 0.8)])
dataset_test = np.array(df[int(df.shape[0] * 0.8):])


def create_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i - 50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x, y


# Now we are going to create our training and testing data by calling our function for each one:

x_train, y_train = create_dataset(dataset_train)
x_test, y_test = create_dataset(dataset_test)

# Next, we need to reshape our data to make it a 3D array in order to use it in LSTM Layer.

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


""" ================================= MODEL BUILDING ================================= """

"""
First, we initialized our model as a sequential one with 96 units in the output’s dimensionality. 
We used return_sequences=True to make the LSTM layer with three-dimensional input and input_shape to shape our dataset.
Making the dropout fraction 0.2 drops 20% of the layers
"""
model = Sequential()
model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=96,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))
model.add(Dropout(0.2))
model.add(Dense(units=1))

"""
After that, we want to reshape our feature for the LSTM layer, because it is sequential_3 which
 is expecting 3 dimensions, not 2:
"""

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

"""
We used loss='mean_squared_error' because it is a regression problem,
 and the adam optimizer to update network weights iteratively based on training data.
"""

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=32) # fit = train
model.save('stock_prediction.h5')

"""
Every epoch refers to one cycle through the full training dataset, 
and batch size refers to the number of training examples utilized in one iteration.
"""

model = load_model('stock_prediction.h5') # load the model
