"""
compute.py
"""
from typing import List
from models import Rating, DiscreteContinuousCoupled
import json

# import algs.diversification as div
import algs.fishingnet as div_fishingnet
import pandas as pd
import numpy as np
import os
import pickle
import time
import csv

def get_predictions(ratings: List[Rating], user_id) -> pd.DataFrame:
    model_file = os.path.join(os.path.dirname(__file__), 'algs/data/explicitMF.pkl')
    f_import = open(model_file, 'rb')
    trained_model = pickle.load(f_import)
    f_import.close()
    
    # new_ratings = pd.Series(rating.rating for rating in ratings)
    # rated_items = np.array([np.int64(rating.item_id) for rating in ratings])
    # The indexing issue was updated on Jan. 28th, 2022
    rated_items = np.array([np.int64(rating.item_id) for rating in ratings])
    new_ratings = pd.Series(np.array([np.float64(rating.rating) for rating in ratings]), index = rated_items)  
    
    items = trained_model.item_index_
    
    als_explicit_preds = trained_model.predict_for_user(user_id, items, new_ratings)
        # return a series with 'items' as the index
    als_explicit_preds_df = als_explicit_preds.to_frame().reset_index()
    als_explicit_preds_df.columns = ['item', 'score']
    
    return als_explicit_preds_df
    
def predict_diverse_items(ratings: List[Rating], user_id) -> pd.DataFrame:
    date_file = os.path.join(os.path.dirname(__file__), 'algs/data/averaged_item_score_explicitMF.csv')
    ave_item_score = pd.read_csv(date_file)
        # ['item', 'ave_score'] 
    
    preds_for_the_user = get_predictions(ratings, user_id)
        # ['item', 'score']
    preds_aveScore = pd.merge(preds_for_the_user, ave_item_score, how = 'left', on = 'item')
        # ['item', 'score', 'ave_score']
    #data_file = os.path.join(os.path.dirname(__file__), 'algs/data/predictions_for_Bart_with_averaged_item_scores_explicitMF.csv')
    #preds_aveScore.to_csv(data_file, index = False)
        
    data_path = os.path.join(os.path.dirname(__file__), './algs/data/item_popularity.csv')    
    item_popularity = pd.read_csv(data_path) 
    
    preds_aveScore_popularity = pd.merge(preds_aveScore, item_popularity, how = 'left', on = 'item')
        # ['item', 'score', 'ave_score', 'count', 'rank']
    cutoffCount = 50
    diversification_candidates = preds_aveScore_popularity[preds_aveScore_popularity['count'] >= cutoffCount]
    # print('\t %d candidates for diversification.' % diversification_candidates.shape[0])
        # ['item', 'score', 'ave_score', 'count', 'rank']

    '''
    numRecs = 80
    items = diversification_candidates['item'].to_numpy()
    feature_vectors = diversification_candidates[['score', 'ave_score']].to_numpy()
    
    # Apply the diversification algorithms
    [recs_diverse_items, _] = div.diversify_item_feature(diversification_candidates, feature_vectors, items, numRecs)
    '''
    numRecs = 80
    # Apply the diversification algorithms
    diverse_items = div_fishingnet.fishingnet_diversification(diversification_candidates, cutoffCount = 50, numRecs = 80)
    
    new_min = 1
    new_max = 5
    [diverse_items_rescared, ave_community_rescared_score, ave_user_rescared_score] = div_fishingnet.scaling_N_labeling(diverse_items, new_min, new_max)
        # diverse_items_rescared: ['item', 'score', 'ave_score', 'count', 'rank', 'community', 'user', 'label_community', 'label_user']
    
    diverse_items = []
    for index, row in diverse_items_rescared.iterrows():
        diverse_items.append(DiscreteContinuousCoupled(str(np.int64(row['item'])), row['community'], row['user'], row['label_community'], row['label_user']))

    # return diverse_items, ave_community_rescared_score, ave_user_rescared_score
    return diverse_items
    # needs to be updated according the the app.py function
        # How to return 3 variables and separate them from a returned tuple?????
    
if __name__ == '__main__':
    fullpath_test = os.path.join(os.path.dirname(__file__), './algs/for_testing/ratings_set6_rated_only_Bart.csv')
    liveUserID = 'Bart'
    ratings_liveUser = pd.read_csv(fullpath_test, encoding='latin1')    
    
    ratings = []
    for index, row in ratings_liveUser.iterrows():
        ratings.append(Rating(row['item'], row['rating']))

    start = time.time()
    #[diversified_recs, ave_community_rescared_score, ave_user_rescared_score] = predict_diverse_items(ratings, liveUserID)
    diversified_recs = predict_diverse_items(ratings, liveUserID)
    end = time.time()
    print('Time spent: %.0f seconds.' % (end - start))
    print(diversified_recs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    