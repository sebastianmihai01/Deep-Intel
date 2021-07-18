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
with open("../dataset/rating.csv", 'r') as f:
    ratings = pd.read_csv(f, parse_dates=['timestamp'])

"""
ratings:

          userId  movieId  rating           timestamp
0              1        2     3.5 2005-04-02 23:53:47
1              1       29     3.5 2005-04-02 23:31:16
2              1       32     3.5 2005-04-02 23:33:39
3              1       47     3.5 2005-04-02 23:32:07
4              1       50     3.5 2005-04-02 23:29:40
...          ...      ...     ...                 ...
20000258  138493    68954     4.5 2009-11-13 15:42:00
"""


# Generates a random sample from a given 1-D array of userId's
rand_userIds = np.random.choice(ratings['userId'].unique(),
                                size=int(len(ratings['userId'].unique()) * 0.3),
                                replace=False)

# Only select data from the random_userIds selected
# loc = Access a group of rows and columns by label(s) or a boolean array
ratings = ratings.loc[ratings['userId'].isin(rand_userIds)]




"""
The code below will split our ratings dataset into a train and test set using the leave-one-out methodology.
"""

# A groupby operation involves some combination of splitting the object, applying a function, and combining the results.
# This can be used to group large amounts of data and compute operations on these groups.

# Compute numerical data ranks (1 through n) along axis.
# first: ranks assigned in order they appear in the array
ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)

train_ratings = ratings[ratings['rank_latest'] != 1]
test_ratings = ratings[ratings['rank_latest'] == 1]

# drop columns that we no longer need
train_ratings = train_ratings[['userId', 'movieId', 'rating']]
test_ratings = test_ratings[['userId', 'movieId', 'rating']]


"""
However, the MovieLens dataset that we are using is based on explicit feedback. To convert this dataset into an implicit 
feedback dataset, we’ll simply binarize the ratings and convert them to ‘1’ (i.e. positive class). 
The value of ‘1’ represents that the user has interacted with the item.
"""

"""
>>> df.loc['cobra':'viper', 'max_speed']
cobra    1
viper    4
Name: max_speed, dtype: int64
"""
train_ratings.loc[:, 'rating'] = 1


"""
Each user will have a number of positive and negative events associated to them. A positive event is one where the user
 bought a movie. A negative event is one where the user saw the movie but decided to not buy.

We do have a problem now though. After binarizing our dataset, we see that every sample in the dataset now belongs to 
the positive class. However, we also require negative samples to train our models, to indicate movies that 
the user has not interacted with. We assume that such movies are those that the user are not interested in — 
even though this is a sweeping assumption that may not be true, it usually works out rather well in practice.

"""


"""
We do have a problem now though. After binarizing our dataset, we see that every sample in the dataset now belongs 
to the positive class

The code below generates 4 negative samples for each row of data. In other words, the ratio of negative to positive 
samples is 4:1. This ratio is chosen arbitrarily but I found that it works rather well in practice(feel free to find
 the best ratio yourself!).
"""

# Get a list of all movie IDs
all_movieIds = ratings['movieId'].unique()

# Placeholders that will hold the training data
users, items, labels = [], [], []

# This is the set of items that each user has interaction with
user_item_set = set(zip(train_ratings['userId'], train_ratings['movieId']))

# 4:1 ratio of negative to positive samples
# consider 4 already existing movies as negatives
num_negatives = 4

for (u, i) in user_item_set:
    users.append(u)
    items.append(i)
    labels.append(1) # items that the user has interacted with are positive
    for _ in range(num_negatives):
        # randomly select an item
        negative_item = np.random.choice(all_movieIds)
        # check that the user has not interacted with this item
        while (u, negative_item) in user_item_set:
            negative_item = np.random.choice(all_movieIds)
        users.append(u)
        items.append(negative_item)
        labels.append(0) # items not interacted with are negative



"""
 Before we move on, let’s define a PyTorch Dataset to facilitate training. The class below simply encapsulates the code 
 we have written above into a PyTorch Dataset class.
 
 See: custom_dataset.py
"""