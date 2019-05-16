from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS
from data import split_data, create_similarity_matrix_cosine, predict_ratings, pivot_ratings
from data import extract_genres, pivot_genres, create_similarity_matrix_categories, mse
from data import extract_elite, get_business, extract_checkin

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
    print(user_id, business_id, city)
    if user_id is not None:
        if city is not None:
            
            bus = DataFrame(BUSINESSES[city])
            reviews = DataFrame(REVIEWS[city])

            business = bus[bus['business_id'] == business_id]
            rating = business['stars'].values[0]

            #Item based ratings
            df_reviews = reviews[['user_id', 'business_id', 'stars']]
            predictions_item_based = prediction_item_based(df_reviews)
            # Hier filter je op predictions groter of gelijk aan de rating van het bedrijf
            mse_item_based = mse(predictions_item_based[predictions_item_based['predicted stars'] >= rating])
            print("mse item based:", mse_item_based)

            #Content based categories
            df_business = bus[['business_id', 'name', 'categories']]
            predictions_content_based = prediction_content_based(df_business, df_reviews)
            # Hier filter je op predictions groter of gelijk aan de rating van het bedrijf
            mse_content_based = mse(predictions_content_based[predictions_content_based['predicted stars'] >= rating])
            print("mse content based:", mse_content_based)

            #Item based based on useful
            user = DataFrame(USERS[city])
            df_user = user[['user_id', 'useful', 'elite']]
            df_useful_users = df_user[df_user['useful'] > 0]
            useful_users = [i for i in df_useful_users['user_id']]

            df_reviews_useful = df_reviews[df_reviews['user_id'].isin(useful_users)]
            predictions_item_based_useful = prediction_item_based(df_reviews_useful)
            mse_item_based_useful = mse(predictions_item_based_useful[predictions_item_based_useful['predicted stars'] >= rating])
            print("revised mse item based useful:", mse_item_based_useful)

            #Item based based on elite
            df_elite = extract_elite(df_user)
            df_elite_users = df_elite[df_elite['elite'].str.contains('^\d{4}$')]
            elite_users = [i for i in df_elite_users['user_id']]

            df_reviews_elite = df_reviews[df_reviews['user_id'].isin(elite_users)]
            predictions_item_based_elite = prediction_item_based(df_reviews_elite)
            mse_item_based_elite = mse(predictions_item_based_elite[predictions_item_based_elite['predicted stars'] >= rating])
            print("revised mse item based elite:", mse_item_based_elite)

            checkins = DataFrame(CHECKINS[city])
            check_ins = extract_checkin(checkins)
            df_checkins = check_ins['date'].groupby(check_ins['business_id']).count()
            top_rated_checkins = df_checkins.nlargest(75)
            top_businesses = [i for i in top_rated_checkins.index]
            
            #Item based on checkins
            df_reviews_checkins = df_reviews[df_reviews['business_id'].isin(top_businesses)]
            predictions_item_based_checkins = prediction_item_based(df_reviews_checkins)
            mse_item_based_checkins = mse(predictions_item_based_checkins[predictions_item_based_checkins['predicted stars'] >= rating])
            print("revised mse item based checkins:", mse_item_based_checkins)

            minimale_mse = min(mse_content_based, mse_item_based, mse_item_based_elite, mse_item_based_useful)

            if mse_item_based == minimale_mse:
                predictions_item_based = prediction_item_based(df_reviews)
                top_10 = predictions_item_based.sort_values(by=['predicted stars'], ascending=False)[:10]
                top_10_samples = [get_business(city, i) for i in top_10['business_id']]
                return top_10_samples
            
            if mse_content_based == minimale_mse:
                predictions_content_based = prediction_content_based(df_business, df_reviews)
                top_10 = predictions_content_based.sort_values(by=['predicted stars'], ascending=False)[:10]
                top_10_samples = [get_business(city, i) for i in top_10['business_id']]
                return top_10_samples
            
            if mse_item_based_useful == minimale_mse:
                predictions_item_based_useful = prediction_item_based(df_reviews_useful)
                top_10 = predictions_item_based_useful.sort_values(by=['predicted stars'], ascending=False)[:10]
                top_10_samples = [get_business(city, i) for i in top_10['business_id']]
                return top_10_samples
            
            if mse_item_based_elite == minimale_mse:
                predictions_item_based_elite = prediction_item_based(df_reviews_elite)
                top_10 = predictions_item_based_elite.sort_values(by=['predicted stars'], ascending=False)[:10]
                top_10_samples = [get_business(city, i) for i in top_10['business_id']]
                return top_10_samples
            
            if mse_item_based_checkins == minimale_mse:
                predictions_item_based_checkins = prediction_item_based(df_reviews_checkins)
                top_10 = predictions_item_based_checkins.sort_values(by=['predicted stars'], ascending=False)[:10]
                top_10_samples = [get_business(city, i) for i in top_10['business_id']]
                return top_10_samples
                
            
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
