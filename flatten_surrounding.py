# -*- coding: utf-8 -*-
'''
This file defines some utility functions to flatten the surrounding database,
which are called by the main script
Author: Cheng CHEN (cheng.chen@nestle.com)
'''


import pandas as pd

def parse_amenity(dict_json):
    '''
    Parse the dictionary (extracted from json), which contains information of a single amenity
    Construct a flat data frame using relevant information.
    The returned data frame has only one row which corresponding to this amenity structure
    '''
    
    dict_tmp = dict()    
    
    # using get() instead of direct indexing as get() will return empty values instead of throwing exceptions
    # if the key does not exist (which can be the case for some fields such as "rating")
    dict_tmp['place_id'] = dict_json.get('place_id')
    dict_tmp['name'] = dict_json.get('name')
    dict_tmp['types'] = ';'.join(dict_json.get('types'))
    rating = dict_json.get('rating')
    if rating != None:
        dict_tmp['rating'] = float(rating)
    rating_total = dict_json.get('user_ratings_total')
    if rating_total != None:
        dict_tmp['rating_total'] = float(rating_total)    
    dict_tmp['longitude'] = dict_json.get('longitude')
    dict_tmp['latitude'] = dict_json.get('latitude')    
    
    # a bit more tricky to get info from the address components
    list_of_dict_address = dict_json['address_components']
    types = [d['types'][0] for d in list_of_dict_address]        
    if 'country' in types:
        dict_tmp['country'] = list_of_dict_address[types.index('country')]['short_name']
    if 'administrative_area_level_1' in types:
        dict_tmp['canton'] = list_of_dict_address[types.index('administrative_area_level_1')]['short_name']
    if 'administrative_area_level_2' in types:
        dict_tmp['commune'] = list_of_dict_address[types.index('administrative_area_level_2')]['short_name']
    if 'locality' in types:
        dict_tmp['locality'] = list_of_dict_address[types.index('locality')]['short_name']       
    
    df_amenity = pd.DataFrame(dict_tmp, index=[0])
    return df_amenity


def flatten_surrounding(df_surrounding):
    '''
    Flatten the surrounding data frame. The original structure is in a nested json structure.
    A flat data frame is constructed using relevant information of amenities, where each row 
    corresponds to one amenity of one store_code.
    
    Construct the long and flat DataFrame, where each POS is spanned in multiple lines, and each line is one surrounding amenity
    For memory efficiency reasons, we construct a list of multiple data frames (each data frame corresponds to one store_code), 
    and concatenate these data frames together finally. This is avoids frequent memory allocations if we only maintained one big data frame
    and append to it one row each time
    '''
    list_df_flat = []
    
    for index in range(len(df_surrounding)):    
        df_flat = pd.DataFrame()
        store_code = df_surrounding['store_code'].iloc[index]
        #print('** Processing store_code {}'.format(store_code))        
        dict_surrounding = df_surrounding['surroundings'].iloc[index]
        keys = dict_surrounding.keys()
        for key in keys:            
            list_dict_amenities = dict_surrounding[key]
            for dict_amenity in list_dict_amenities:            
                df_amenity = parse_amenity(dict_amenity)
                df_amenity['store_code'] = store_code
                df_amenity['category'] = key
                df_flat = df_flat.append(df_amenity)               
        list_df_flat.append(df_flat)
    
    df_surrounding = pd.concat(list_df_flat)
    
    # rearange columns
    cols = ['store_code','category','place_id','name','country','canton','commune','locality','longitude','latitude','rating','rating_total','types']
    df_surrounding = df_surrounding[cols]
    return df_surrounding