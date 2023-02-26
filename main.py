import random
import numpy as np
import pandas as pd
import sklearn
# import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.linalg import sqrtm
import math
import statistics

df = pd.read_csv ('user_artist.csv')
df1 = pd.read_csv ('test.csv')
df['userID'] = df['userID'].astype('str')
df['artistID'] = df['artistID'].astype('str')
train = df.copy()
test = df1.copy()
#train, test = sklearn.model_selection.train_test_split(df, train_size = 0.8)

#calculation of prediction for users from test apearing only once in train
# and artists from test apearing only once in train
df_lonley= df.groupby('userID').size().reset_index()
df_lonley.rename(columns = {0:'count'},inplace = True)
dflonley1 = df_lonley.loc[df_lonley['count'] == 1]
list_of_singles = dflonley1['userID'].tolist()
singler_filter = df['userID'].isin(list_of_singles)
full_single = df[singler_filter]
lon_user = full_single['weight'].tolist()
pred_for_lon_user = statistics.median(lon_user)

df_lonley_artist= df.groupby('artistID').size().reset_index()
df_lonley_artist.rename(columns = {0:'count'},inplace = True)
dflonley_artist1 = df_lonley_artist.loc[df_lonley_artist['count'] == 1]
list_of_singles_artists = dflonley_artist1['artistID'].tolist()
singler_filter_art = df['artistID'].isin(list_of_singles_artists)
full_single_art = df[singler_filter_art]
lon_artist = full_single_art['weight'].tolist()
pred_for_lon_artist = statistics.median(lon_artist)

#from dataframe to matrix
def create_utility_matrix(data, formatizer={'user': 0, 'item': 1, 'value': 2}):
    """
        :param data:      Array-like, 2D, nx3
        :param formatizer:pass the formatizer
        :return:          utility matrix (n x m), n=users, m=items
    """

    itemField = formatizer['item']
    userField = formatizer['user']
    valueField = formatizer['value']
    userList = data.iloc[:,  userField].tolist()
    # userList = data['userID'].drop_duplicates().tolist()
    itemList = data.iloc[:,  itemField].tolist()
    valueList = data.iloc[:, valueField].tolist()
    users = list(set(data.iloc[:, userField]))
    items = list(set(data.iloc[:, itemField]))
    users_index = {users[i]: i for i in range(len(users))}
    pd_dict = {item: [np.nan for i in range(len(users))] for item in items}
    for i in range(0, len(data)):
        item = itemList[i]
        user = userList[i]
        value = valueList[i]
        pd_dict[item][users_index[user]] = value
    X = pd.DataFrame(pd_dict)
    X.index = users

    itemcols = list(X.columns)
    items_index = {itemcols[i]: i for i in range(len(itemcols))}
    # users_index gives us a mapping of user_id to index of user
    # items_index provides the same for items
    return X, users_index, items_index

#calculating SVD matrix
def svd(train, k):
    utilMat = np.array(train)
    # the nan or unavailable entries are masked
    mask = np.isnan(utilMat)
    masked_arr = np.ma.masked_array(utilMat, mask)
    item_means = np.mean(masked_arr, axis=0)
    # nan entries will replaced by the average rating for each item
    utilMat = masked_arr.filled(item_means)
    x = np.tile(item_means, (utilMat.shape[0],1))
    # we remove the per item average from all entries.
    # the above mentioned nan entries will be essentially zero now
    utilMat = utilMat - x
    # The magic happens here. U and V are user and item features
    U, s, V=np.linalg.svd(utilMat, full_matrices=False)
    s=np.diag(s)
    # we take only the k most significant features
    s=s[0:k,0:k]
    U=U[:,0:k]
    V=V[0:k,:]
    s_root=sqrtm(s)
    Usk=np.dot(U,s_root)
    skV=np.dot(s_root,V)
    UsV = np.dot(Usk, skV)
    UsV = UsV + x
    print("svd done")
    return UsV

#calculating RMSE. used for training the model.
def rmse(true, pred):
    # this will be used towards the end
    x = (np.log(true) - np.log(pred))**2
    return sum(x),pred


# to test the performance over a different number of features
no_of_features = [10]#to choose K

utilMat, users_index, items_index = create_utility_matrix(train)

#all together
for f in no_of_features:
    svdout = svd(utilMat, k=f)
    # print(svdout)
    pred = [] #to store the predicted ratings
    for _,row in test.iterrows():
        user = str(row['userID'])
        item = str(row['artistID'])
        if str(item) not in train['artistID'].unique():
            pred_rating = pred_for_lon_artist
            print("lon art loop: " + str(item))
        elif str(user) not in train['userID'].unique():
            pred_rating = pred_for_lon_user
            print("lon user loop: " + str(user))
        #if str(item) in train['artistID'].unique() and str(user) in train['userID'].unique():
        else:
            u_index = users_index[user]
            i_index = items_index[item]
            pred_rating = svdout[u_index, i_index]
        pred.append(pred_rating)
    print(pred)


df1['weight'] = pred
df1.to_csv("test_weight2.csv", encoding='utf-8', index=False)
