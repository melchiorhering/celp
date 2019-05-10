from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS

import random
import json
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
np.seterr(divide='raise', over='raise', under='raise', invalid='raise')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D


def recommend(user_id=None, business_id=None, city=None, n=10):
    """
    Returns n recommendations as a list of dicts.
    Optionally takes in a user_id, business_id and/or city.
    A recommendation is a dictionary in the form of:
        {
            business_id:str
            stars:str
            name:str
            city:str
            adress:str
        }
    """
    if not city:
        city = random.choice(CITIES)
    return random.sample(BUSINESSES[city], n)

def extract_genres(bedrijven):
    """Create an unfolded genre dataframe. Unpacks genres seprated by a '|' into seperate rows.

    Arguments:
    movies -- a dataFrame containing at least the columns 'movieId' and 'genres' 
              where genres are seprated by '|'
    """
    categories_b = bedrijven.apply(lambda row: pd.Series([row['business_id']] + row['categories'].lower().split(",")), axis=1)
    stack_categories = categories_b.set_index(0).stack()
    df_stack_categories = stack_categories.to_frame()
    df_stack_categories['business_id'] = stack_categories.index.droplevel(1)
    df_stack_categories.columns = ['categories', 'business_id']
    return df_stack_categories.reset_index()[['business_id', 'categories']]

def pivot_genres(df):
    """Create a one-hot encoded matrix for genres.
    
    Arguments:
    df -- a dataFrame containing at least the columns 'movieId' and 'genre'
    
    Output:
    a matrix containing '0' or '1' in each cell.
    1: the movie has the genre
    0: the movie does not have the genre
    """
    return df.pivot_table(index = 'business_id', columns = 'categories', aggfunc = 'size', fill_value=0)

def create_similarity_matrix_categories(matrix):
    """Create a  """
    npu = matrix.values
    m1 = npu @ npu.T
    diag = np.diag(m1)
    m2 = m1 / diag
    m3 = np.minimum(m2, m2.T)
    return pd.DataFrame(m3, index = matrix.index, columns = matrix.index)

def predict_ratings(similarity, utility, to_predict):
    """Predicts the predicted rating for the input test data.
    
    Arguments:
    similarity -- a dataFrame that describes the similarity between items
    utility    -- a dataFrame that contains a rating for each user (columns) and each business (rows). 
                  If a user did not rate an item the value np.nan is assumed. 
    to_predict -- A dataFrame containing at least the columns business_id and userId for which to do the predictions
    """
    # copy input (don't overwrite)
    ratings_test_c = to_predict.copy()
    # apply prediction to each row
    ratings_test_c['predicted rating'] = to_predict.apply(lambda row: predict_ids(similarity, utility, row['userId'], row['business_id']), axis=1)
    return ratings_test_c

### Helper functions for predict_ratings_item_based ###

def predict_ids(similarity, utility, userId, itemId):
    # select right series from matrices and compute
    if userId in utility.columns and itemId in similarity.index:
        return predict_vectors(utility.loc[:,userId], similarity[itemId])
    return 0

def predict_vectors(user_ratings, similarities):

    # select only movies actually rated by user
    relevant_ratings = user_ratings.dropna()
    
    # select corresponding similairties
    similarities_s = similarities[relevant_ratings.index]
    
    # select neighborhood
    similarities_s = similarities_s[similarities_s > 0.0]
    relevant_ratings = relevant_ratings[similarities_s.index]
    
    # if there's nothing left return a prediction of 0
    norm = similarities_s.sum()
    if(norm == 0):
        return 0
    
    # compute a weighted average (i.e. neighborhood is all) 
    return np.dot(relevant_ratings, similarities_s)/norm


def get_rating(ratings, userId, businessId):
    """Given a userId and businessId, this function returns the corresponding rating.
       Should return NaN if no rating exists."""
    rating = ratings.loc[(ratings['user_id'] == userId) & (ratings['business_id'] == businessId)]

    print(rating)
    return rating['stars'].item()

def pivot_data(matrix):

    business_ids = matrix['business_id'].unique()
    user_ids = matrix['user_id'].unique()
    pivot_data = DataFrame(matrix, index=business_ids, columns=user_ids)

    for u_id in user_ids:
        for m_id in business_ids:
            pivot_data[m_id, u_id] = get_rating(matrix, u_id, m_id) 
    return pivot_data


# def pivot_ratings(ratings):
#     """ takes a rating table as input and computes the utility matrix """
#     # get business and user id's
#     business_ids = ratings['business_id'].unique()
#     user_ids = ratings['user_id'].unique()
#     # create empty data frame
#     pivot_data = pd.DataFrame(ratings, columns=business_ids, index=user_ids, dtype=float)
#     # use the function get_rating to fill the matrix
#      for u_id in userIds:
#         for m_id in movieIds:
#             pivot_data.loc[m_id, u_id] = get_rating(ratings, u_id, m_id) 
#     return pivot_data







# Dataframe van bedrijven van een stad
bedrijf = DataFrame(BUSINESSES['westlake'])
# Dataframe van reviews van gebruikers per bedrijf in stad
user_reviews = DataFrame(REVIEWS['westlake'])
# Dataframe van users per stad
users = DataFrame(USERS['westlake'])

"""
Similarity matrix aan de hand van categorieën
""" 
# Zet de catogries los van de bedrijven
df_categories = extract_genres(bedrijf)
# print(df_categories[:10])

# Pivot de dataframe
df_utility_categories = pivot_genres(df_categories)
# print(df_utility_categories.head())

# Maak similarity Matrix
df_similarity_categories = create_similarity_matrix_categories(df_utility_categories)
# print(df_similarity_categories.head())

"""END"""


""" Similarity matrix aan de hand van rating van gebruiker per bedrijf """

# Filter user_reviews op business_id, review_id, stars en useful
user_reviews_filtered = DataFrame(user_reviews, columns=['business_id', 'user_id','stars', 'useful'])
print(user_reviews_filtered.head())

pivot_data(user_reviews_filtered)


