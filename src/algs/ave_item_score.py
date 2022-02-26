import sys
import warnings
if not sys.warnoptions:
        warnings.simplefilter("ignore")
'''
    There will be NumbaDeprecationWarnings here, use the above code to hide the warnings
'''            
import numpy as np
import pandas as pd
from lenskit.algorithms import als
import setpath
import time
import pickle
from load_npz import load_trainset_npz
import os
    
def averaged_item_score(algo, transet):  
    '''
        algo: trained implicitMF model
        transet: ['user', 'item', 'rating', 'timestamp']
    '''
    ###
    items = transet.item.unique()
        # items is NOT sorted by derault
    users = transet.user.unique()
    num_users = len(users)
        # users is NOT sorted by derault
    
    ## items: ndarray -> df
    ave_scores_df = pd.DataFrame(items, columns = ['item'])
    ave_scores_df['ave_score'] = 0
    calculated_users = -1
    start = time.time()
    for user in users:
    #for user in users[0:1000]:
        calculated_users += 1;
        print(num_users - (calculated_users + 1), end = '\r') 
            # flushing does not work
        user_implicit_preds = algo.predict_for_user(user, items)
        # print(type(user_implicit_preds))
            # the ratings of the user is already in the trainset used to train the algo
            # return a series with 'items' as the index, order is the same
        user_implicit_preds_df = user_implicit_preds.to_frame().reset_index()
        user_implicit_preds_df.columns = ['item', 'score']
                
        ave_scores_df['ave_score'] = (ave_scores_df['ave_score'] * calculated_users + user_implicit_preds_df['score'])/(calculated_users + 1)
    print("\nIt took %.0f seconds to calculate the averaved item scores." % (time.time() - start))
    # 1095s
    
    return ave_scores_df
    
if __name__ == "__main__":    
    
    
    ### Import explicit MF model, saved in an object
    data_path = setpath.set_working_path()
    model_filename = os.path.join(data_path, 'explicitMF.pkl')
    f_import = open(model_filename, 'rb')
    algo = pickle.load(f_import)
    f_import.close()
    
    ### Import offline dataset, this was  also used as the transet in RSSA
    fullpath_trian = os.path.join(data_path, 'train.npz')
    #fullpath_trian = data_path + 'train.npz'
    attri_name = ['user', 'item', 'rating', 'timestamp']
    ratings_train = load_trainset_npz(fullpath_trian, attri_name)
    
    ave_item_score = averaged_item_score(algo, ratings_train)
    ave_item_score.to_csv(data_path + 'averaged_item_score_explicitMF.csv', index = False)