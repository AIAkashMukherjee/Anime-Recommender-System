from config.path_config import *
from utils.helper import *


def hybrid_recommendation(user_id , user_weight=0.6, content_weight =0.4):
     ## User Recommndation
     similar_users =find_similar_users(user_id,USER_WEIGHTS_PATH,USER2USER_ENCODED,USER2USER_DECODED)
     user_pref = get_user_preferences(user_id , RATING_DF, DF)
     user_recommended_animes =get_user_recommendations(similar_users,user_pref,DF, SYNOPSIS_DF,RATING_DF)

     user_recommended_anime_list = user_recommended_animes["anime_name"].tolist()

     ## Content recommendation
     content_recommendation_animes=[]
     for anime in user_recommended_anime_list:
          similar_animes = item_based_recommnedation_system(anime, ANIME_WEIGHTS_PATH, ANIME2ANIME_ENCODED, ANIME2ANIME_DECODED,DF)
          if similar_animes is not None and not similar_animes.empty:
               content_recommendation_animes.extend(similar_animes['name'].tolist())
               
          else:
            print(f"No similar anime found {anime}")
          
     # combing scores   
     combined_scores = {}   

     for anime in user_recommended_anime_list:
         combined_scores[anime]= combined_scores.get(anime,0) + user_weight

     for anime in content_recommendation_animes:
        combined_scores[anime] = combined_scores.get(anime,0) + content_weight


     sorted_animes=sorted(combined_scores.items(),key=lambda x:x[1], reverse=True)

     return [anime for anime , score in sorted_animes[:10]] 

     
