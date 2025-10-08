import pandas as pd
import numpy as np
import joblib
from config.path_config import *


def getAnimeFrame(anime,path_df):
    df = pd.read_csv(path_df)
    if isinstance(anime,int):
        return df[df['anime_id']==anime]
    if isinstance(anime,str):
        return df[df['eng_version'] == anime]
    
 

def getsynopsis(anime,path_synopsis_df):
    synopsis_df = pd.read_csv(path_synopsis_df)
    if isinstance(anime,int):
        return synopsis_df[synopsis_df.MAL_ID == anime].sypnopsis.values[0]
    if isinstance(anime,str):
        return synopsis_df[synopsis_df.Name == anime].sypnopsis.values[0]    
     



def item_based_recommnedation_system(name, path_anime_weights, path_anime2anime_encoded, path_anime2anime_decoded, df, n=10, return_dist=False,neg=False):
    try:
        anime_weights = joblib.load(path_anime_weights)
        anime2anime_encoded = joblib.load(path_anime2anime_encoded)
        anime2anime_decoded = joblib.load(path_anime2anime_decoded)

        index = getAnimeFrame(name,df).anime_id.values[0]
        encoded_index=anime2anime_encoded.get(index)

        weights=anime_weights

        dist=np.dot(weights,weights[encoded_index])
        sorted_dists = np.argsort(dist)

        n=n+1
        if neg: # will fetch dismiar animes
            closest=sorted_dists[:n]
            # print(f'Dissimilar anime to {name}')

        else: # this will fetch similar animes
            closest = sorted_dists[-n:] 

            # print(f'Similar anime to {name}')

        if return_dist:
            return dist,closest             
    
        similarlityArr=[]
        for close in closest:
            decoded_id=anime2anime_decoded.get(close)

            anime_frame=getAnimeFrame(decoded_id,df)

            anime_name=anime_frame.eng_version.values[0]
            genre=anime_frame.Genres.values[0]
            score=anime_frame.Score.values[0]
            similarity = dist[close]

            similarlityArr.append({
            "anime_id": decoded_id,
            "name": anime_name,
            "similarity": similarity,
            "genre": genre,
            'Score':score,
            })


        Frame=pd.DataFrame(similarlityArr).sort_values(by="similarity", ascending=False)
        return Frame[Frame.anime_id!=index].drop(['anime_id'], axis=1)
    
    except Exception as e:
        print("Error Occured",e)


def find_similar_users(item_input , path_user_weights , path_user2user_encoded , path_user2user_decoded, n=10 , return_dist=False,neg=False):
    try:
        user_weights = joblib.load(path_user_weights)
        user2user_encoded = joblib.load(path_user2user_encoded)
        user2user_decoded = joblib.load(path_user2user_decoded)

        index=item_input
        encoded_index=user2user_encoded.get(index)

        weights = user_weights

        dists = np.dot(weights,weights[encoded_index])
        sorted_dists = np.argsort(dists)

        n=n+1

        if neg:
            closest = sorted_dists[:n]
            
        else:
            closest = sorted_dists[-n:]

        if return_dist:
            return dists,closest             
    
        SimilarityArr=[]    
        for close in closest:
            similarity = dists[close] 
            if isinstance(item_input,int):
                decoded_idx=user2user_decoded.get(close)
                SimilarityArr.append({
                    "similar_users" : decoded_idx,
                    "similarity" : similarity
                })
        similar_users = pd.DataFrame(SimilarityArr).sort_values(by="similarity",ascending=False)
        similar_users = similar_users[similar_users.similar_users != item_input]
        return similar_users   

    except Exception as e:
        print("Error Occured",e)    


# extract user perfences

def get_user_preferences(user_id , path_rating_df , path_anime_df ):
    rating_df = pd.read_csv(path_rating_df)
    df = pd.read_csv(path_anime_df)

    animes_watched_by_user = rating_df[rating_df.user_id==user_id]

    user_rating_percentile = np.percentile(animes_watched_by_user.rating,70)
    animes_watched_by_user = animes_watched_by_user[animes_watched_by_user.rating>=user_rating_percentile]
    top_anime_user=(
        animes_watched_by_user.sort_values(by="rating" , ascending=False).anime_id.values
    )

    anime_df_Rows=df[df['anime_id'].isin(top_anime_user)]
    anime_df_Rows=anime_df_Rows[["eng_version","Genres"]]

    return anime_df_Rows


def get_user_recommendations(similar_users , user_pref ,df , synopsis_df, rating_df, n=10):
    recommended_animes = []
    anime_list = []

    for user_id in similar_users.similar_users.values:
        # gettting user preference of similar users
        pref_list=get_user_preferences(int(user_id),rating_df,df)
        # exclude already rated names from this prefences
        pref_list=pref_list[~pref_list.eng_version.isin(user_pref.eng_version.values)]

        if not pref_list.empty:
            anime_list.append(pref_list.eng_version.values)

    if anime_list:
        anime_list = pd.DataFrame(anime_list)    
        # we have got top n anime in this sorted list
        sorted_list=pd.DataFrame(pd.Series(anime_list.values.ravel()).value_counts()).head(n)
        # print(sorted_list)

        for i , anime_name in enumerate(sorted_list.index):
            n_user_pref = sorted_list[sorted_list.index == anime_name].values[0][0]

            # if anime name is in string
            if isinstance(anime_name,str):
                frame=getAnimeFrame(anime_name,df)
                anime_id = frame.anime_id.values[0]
                genre = frame.Genres.values[0]
                score=frame.Score.values[0]
                synopsis = getsynopsis(int(anime_id),synopsis_df)

                recommended_animes.append({
                    "No_of_user_prefences" : n_user_pref,
                    'Scores': score,
                    "anime_name" : anime_name,
                    "Genres" : genre,
                    "Synopsis": synopsis
                })
    return pd.DataFrame(recommended_animes).head(n)