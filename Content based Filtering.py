#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


# Load the Spotify dataset
df = pd.read_csv('top_tracks.csv', header=0)
df


# In[3]:


df.columns


# In[4]:


df.isna().sum()


# In[5]:


df.dropna(how='any', inplace=True)


# In[6]:


df.isna().sum()


# In[7]:


# Select the relevant audio features
audio_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']


# In[8]:


# Normalize the audio feature data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df[audio_features])


# In[9]:


# Calculate the similarity between songs
similarity_matrix = cosine_similarity(normalized_data)


# In[10]:


'''def recommend_songs(name, no_of_recommendations):
    #get the index of the game that matches the title
    idx = df[df['name'] == name].index[0]

    #create a series with the similarity scores in descending order
    score_series = pd.Series(similarity_matrix[idx]).sort_values(ascending=False)

    #get the indexes of the most similar games
    top_indexes = list(score_series.iloc[1:no_of_recommendations+1].index)

    #return most similar games
    return df.iloc[top_indexes]'''

def recommend_songs(name, no_of_recommendations):
    #get the index of the game that matches the title
    idx = df[df['name'] == name].index[0]

    #create a series with the similarity scores in descending order
    score_series = pd.Series(similarity_matrix[idx]).sort_values(ascending=False)

    #get the indexes of the most similar games
    top_indexes = list(score_series.iloc[1:no_of_recommendations+1].index)

    #get the unique list of artists from the recommended songs
    artist_list = list(set(df.iloc[top_indexes]['artist']))

    #return most similar games and list of artists
    return df.iloc[top_indexes], artist_list


# In[11]:


# Test the recommender system
recommendations, artist_list = recommend_songs('Wish You Were Here', no_of_recommendations=5)
print(recommendations)


# In[12]:


print(artist_list)


# In[27]:


# Load the data
artists = pd.read_csv('artists.dat', delimiter='\t', usecols=['id', 'name'])
user_artists = pd.read_csv('user_artists.dat', delimiter='\t', usecols=['userID', 'artistID', 'weight'])

# Map artist IDs to names
artist_dict = dict(zip(artists['id'], artists['name']))

# Create user-artist matrix
user_artist_matrix = pd.pivot_table(user_artists, values='weight', index='userID', columns='artistID', fill_value=0)

def get_artist_recommendations(artist_list):
    # Create a boolean vector for the input artists
    input_vector = np.zeros(len(artist_dict))
    for artist in artist_list:
        if artist in artist_dict.values():
            artist_id = list(artist_dict.keys())[list(artist_dict.values()).index(artist)]
            input_vector[user_artist_matrix.columns.get_loc(artist_id)] = 1
    
    # Compute similarity between input vector and all artists in the dataset
    sim_vector = cosine_similarity(user_artist_matrix, input_vector.reshape(1, -1)).flatten()
    
    # Get indices of artists with highest similarity values
    sim_indices = sim_vector.argsort()[::-1]
    
    # Get top 10 recommended artists
    recommended_artists = []
    for i in sim_indices:
        if artist_dict[user_artist_matrix.columns[i]] not in artist_list:
            recommended_artists.append(artist_dict[user_artist_matrix.columns[i]])
        if len(recommended_artists) >= 10:
            break
    
    return recommended_artists

# Test the function with an example input
recommendations = get_artist_recommendations(artist_list)
print('Recommendations:', recommendations)

# Print artist dictionary for debugging
#print('Artist dictionary:', artist_dict)


# In[13]:


import streamlit as st
# Set up the Streamlit app
st.set_page_config(page_title='Spotifynd', page_icon=':musical_note:', layout='wide')

st.title('Spotifynd')
st.write('Welcome to Spotifynd, your personal music recommendation engine!')

# Create a sidebar with user input fields
with st.sidebar:
    st.write('Enter the name of a song to find similar songs:')
    song_name = st.text_input('Song name', 'Wish You Were Here')
    st.write('Enter the number of recommendations to show:')
    num_recommendations = st.slider('Number of recommendations', 1, 10, 5)

# Generate recommendations based on user input
recommendations = recommend_songs(song_name, num_recommendations)

# Display the recommendations in a table
st.write('Here are your recommendations:')
st.table(recommendations.style.background_gradient(cmap='viridis').hide_index())


# In[ ]:




