from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS
from data import split_data, create_similarity_matrix_cosine, predict_ratings, pivot_ratings
from data import extract_genres, pivot_genres, create_similarity_matrix_categories, mse
from data import extract_elite, get_business, prediction_item_based, prediction_content_based

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
import re
import spacy
import sklearn
import os
from IPython.display import clear_output

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
    if user_id is not None:
        if city is not None:
            
            # load the datasets
            business = DataFrame(BUSINESSES[city])
            reviews = DataFrame(REVIEWS[city])
            user = DataFrame(USERS[city])

            # get the rating from the business that is clicked on
            businessid = business[business['business_id'] == business_id]
            rating = businessid['stars'].values[0]

            # get the useful users (with a minimum useful review of 1)
            df_user = user[['user_id', 'useful', 'elite']]
            df_useful_users = df_user[df_user['useful'] > 0]
            useful_users = [i for i in df_useful_users['user_id']]

            # get the elite users (with a minimum of 1 year)
            df_elite = extract_elite(df_user)
            df_elite_users = df_elite[df_elite['elite'].str.contains('^\d{4}$')]
            elite_users = [i for i in df_elite_users['user_id']]
            
            # Item based reviews without filter 
            df_reviews = reviews[['user_id', 'business_id', 'stars', 'review_id', 'useful']]
            
            first_training, first_test = split_data(df_reviews, d=0.9)
            first_utility = pivot_ratings(first_training)
            first_similarity = create_similarity_matrix_cosine(first_utility)
            first_predictions = predict_ratings(first_similarity, first_utility, first_test[['user_id', 'business_id', 'stars']])
            first_mse_item = mse(first_predictions[first_predictions['predicted stars'] > rating])
            print("first mse item:", first_mse_item)
            
            # Item based reviews with filter useful users
            df_reviews_useful = df_reviews[df_reviews['user_id'].isin(useful_users)]

            second_training, second_test = split_data(df_reviews_useful, d=0.9)
            second_utility = pivot_ratings(second_training)
            second_similarity = create_similarity_matrix_cosine(second_utility)
            second_predictions = predict_ratings(second_similarity, second_utility, second_test[['user_id', 'business_id', 'stars']])
            second_mse_item = mse(second_predictions[second_predictions['predicted stars'] >= rating])
            print("second mse item:", second_mse_item)

            # Item based reviews with filters useful users and elite users
            df_reviews_elite = df_reviews_useful[df_reviews_useful['user_id'].isin(elite_users)]
            
            third_training, third_test = split_data(df_reviews_elite, d=0.9)
            third_utility = pivot_ratings(third_training)
            third_similarity = create_similarity_matrix_cosine(third_utility)
            third_predictions = predict_ratings(third_similarity, third_utility, third_test[['user_id', 'business_id', 'stars']])
            third_mse_item = mse(third_predictions[third_predictions['predicted stars'] >= rating])
            print("third mse item:", third_mse_item)

            # Content based on categories
            df_business_content = business[['business_id', 'name', 'categories']]

            predictions_content_based_categories = prediction_content_based(df_business_content, df_reviews)
            mse_content_based_categories = mse(predictions_content_based_categories[predictions_content_based_categories['predicted stars'] >= rating])
            print("mse content based categories:", mse_content_based_categories)

            # content based on categories and filters of useful ratings and open businesses
            df_useful_reviews = df_reviews[ (df_reviews['stars'] >= rating - 0.5) & (df_reviews['useful'] > 0)]
             
            useful_reviews_ids = [i for i in df_useful_reviews['review_id']]

            filtered_df_reviews = df_reviews[df_reviews['review_id'].isin(useful_reviews_ids)]

            df_business = business[['business_id', 'is_open', 'stars']]

            open = businessid['is_open'].values[0]
            df_open_businesses = df_business[df_business['is_open'] == open]
            open_businesses = [i for i in df_open_businesses['business_id']]

            filtered_df_businesses_open = filtered_df_reviews[filtered_df_reviews['business_id'].isin(open_businesses)]

            predictions_content_based_categories2 = prediction_content_based(df_business_content, filtered_df_businesses_open)
            mse_content_based_categories2 = mse(predictions_content_based_categories2[predictions_content_based_categories2['predicted stars'] >= rating])
            print("mse content based categories2:", mse_content_based_categories2)

            minimale_mse = min(first_mse_item, second_mse_item, third_mse_item, mse_content_based_categories, mse_content_based_categories2)
            
            if first_mse_item == minimale_mse:
                return_samples(first_predictions, city, business_id)
            
            elif second_mse_item == minimale_mse:
                return_samples(second_predictions, city, business_id)
            
            elif third_mse_item == minimale_mse:
                return_samples(third_predictions, city, business_id)
            
            elif mse_content_based_categories == minimale_mse:
                return_samples(predictions_content_based_categories, city, business_id)

            elif mse_content_based_categories == minimale_mse:
                return_samples(predictions_content_based_categories2, city, business_id)
                
    if not city:
        city = random.choice(CITIES)
    return random.sample(BUSINESSES[city], n)

def return_samples(prediction, city, business_id):
    predictions = prediction[prediction['business_id'] != business_id]
    top_10 = predictions.sort_values(by=['predicted stars'], ascending=False)[:10]
    top_10_samples = [get_business(city, i) for i in top_10['business_id']]
    # print(top_10)
    # print(len(predictions))
    return top_10_samples   