from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS

import random
import json
from pandas import Series
import pandas as pd
from pandas import DataFrame


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


# Dataframe van bedrijven van een stad
b_p_s = DataFrame(BUSINESSES['westlake'])
# Zet de bussines_id als index in deze lijst
b_p_s = b_p_s.set_index('business_id')
# Maak nieuwe df met belangrijke columnen
df = pd.DataFrame(b_p_s,columns=['categories','attributes', 'review_count', 'stars'])
print(df[:10])


