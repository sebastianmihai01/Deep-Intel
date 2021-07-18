"""
Before we start building and training our model, let’s do some preprocessing
to get the MovieLens data in the required format.
"""


"""
In order to keep memory usage manageable, we will only use data from 30% of the users in this dataset.
 Let’s randomly select 30% of the users and only use data from the selected users.
"""

import pandas as pd
import numpy as np

# adds offset so that the numbers will be generated randomly every time
# np.random.seed(0) => same generation every time
np.random.seed(123)

# list of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3 each as a separate date column.
ratings = pd.read_csv('rating.csv', parse_dates=['timestamp'])


rand_userIds = np.random.choice(ratings['userId'].unique(),
                                size=int(len(ratings['userId'].unique()) * 0.3),
                                replace=False)

ratings = ratings.loc[ratings['userId'].isin(rand_userIds)]
