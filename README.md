# Autonomous Logistics, Statistics and Marketing Analysis Model for Your Business Using Deep-Learning

## Dataset

Context
The datasets describe ratings and free-text tagging activities from MovieLens, a movie recommendation service. It contains 20000263 ratings and 465564 tag applications across 27278 movies. These data were created by 138493 users between January 09, 1995 and March 31, 2015. This dataset was generated on October 17, 2016.
Users were selected at random for inclusion. All selected users had rated at least 20 movies.

## Data processing

Before we start building and training our model, let’s do some preprocessing to get the MovieLens data in the required format.
In order to keep memory usage manageable, we will only use data from 30% of the users in this dataset. Let’s randomly select 30% of the users and only use data from the selected users.

## Training and Evaluation

This train-test split strategy is often used when training and evaluating recommender systems. Doing a random split would not be fair, as we could potentially be using a user’s recent reviews for training and earlier reviews for testing. This introduces data leakage with a look-ahead bias, and the performance of the trained model would not be generalizable to real-world performance.
