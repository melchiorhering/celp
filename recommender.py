from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS
from data import split_data, create_similarity_matrix_cosine, predict_ratings, pivot_ratings
from data import extract_genres, pivot_genres, create_similarity_matrix_categories, mse

import random
import json
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
np.seterr(divide='raise', over='raise', under='raise', invalid='raise')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import sklearn.metrics.pairwise as pw

# Content based gebruiken indien er weinig ratings beschikbaar zijn.

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
    print(user_id, business_id, city)
    if user_id is not None:
        if city is not None:
            
            a = DataFrame(BUSINESSES[city])
            b = a[a['business_id'] == business_id]
            rating = b['stars'].values[0]

            #Item based
            reviews = DataFrame(REVIEWS[city])
            df_reviews = reviews[['user_id', 'business_id', 'stars']]
            predictions_item_based = prediction_item_based(df_reviews)

            # Hier filter je op predictions groter of gelijk aan de rating van het bedrijf
            mse_item_based = mse(predictions_item_based[predictions_item_based['predicted stars'] >= rating])

            print(mse_item_based)

            #Content based
            business = DataFrame(BUSINESSES[city])
            df_business = business[['business_id', 'name', 'categories']]
            predictions_content_based = prediction_content_based(df_business, df_reviews)

            # Hier filter je op predictions groter of gelijk aan de rating van het bedrijf
            mse_content_based = mse(predictions_content_based[predictions_content_based['predicted stars'] >= rating])
            
            print(mse_content_based)
            

    if not city:
        city = random.choice(CITIES)
        # print(random.sample(BUSINESSES[city], n))
    return random.sample(BUSINESSES[city], n)

def prediction_item_based(df):
    df_stars_training, df_stars_test = split_data(df, d=0.9)
    df_utility_stars = pivot_ratings(df_stars_training)
    df_similarity_stars = create_similarity_matrix_cosine(df_utility_stars)
    return predict_ratings(df_similarity_stars, df_utility_stars, df_stars_test[['user_id', 'business_id', 'stars']])

def prediction_content_based(df_business, df_reviews):
    df_stars_training, df_stars_test = split_data(df_reviews, d=0.9)
    df_utility_stars = pivot_ratings(df_stars_training)

    df_genres = extract_genres(df_business)
    df_utility_genres = pivot_genres(df_genres)
    df_similarity_genres = create_similarity_matrix_categories(df_utility_genres)
    return predict_ratings(df_similarity_genres, df_utility_stars, df_stars_test[['user_id', 'business_id', 'stars']])



# df_users = DataFrame(USERS[city])
# df_most_rated = df_users[df_users['review_count'] > 10]
# most_rated_users = [i for i in df_most_rated['user_id']] 



# c = a[a['stars'] >= rating]
# d = c[c['review_count'] > 10] 
# e = [i for i in d['business_id']]


# y = x[['user_id', 'business_id', 'stars']]
# z = y[y['business_id'].isin(e)]
# h = z[z['user_id'].isin(most_rated_users)]

