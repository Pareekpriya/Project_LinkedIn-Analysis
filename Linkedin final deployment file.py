# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 22:00:33 2023

@author: user
"""

import pickle
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
#loaded the saved model
cosine_similarity= pickle.load(open("C:/Users/user/Downloads/cosine_similarity.pkl", 'rb'))
jobs_df=pickle.load(open("C:/Users/user/Downloads/jobs_df.pkl",'rb'))


st.set_page_config(layout="centered")

def welcome():
    return "Welcome All"
    return "Welcome All"


# Define the similarity metric
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(jobs_df['Job_Name'])
# Build the recommendation model
def get_recommendations(job_title, num_recommendations):
    job_idx = jobs_df[jobs_df['Job_Name'] == job_title].index[0]
    cosine_similarities = cosine_similarity(tfidf_matrix[job_idx], tfidf_matrix)
    similar_jobs_indices = cosine_similarities.argsort()[0][int(num_recommendations)-1:0:-1]
    return jobs_df.iloc[similar_jobs_indices]



# Define the app
st.title('Jobs Recommender System')

# Get user input
option = st.selectbox('Select your Job: ', jobs_df['Job_Name'].values)
num_recommendations = st.slider("Select the number of recommendations", min_value=1, max_value=50, value=10, step=1)

# Generate recommendations
if st.button('Click here to Recommend'):
    recommendation = get_recommendations(option, num_recommendations)
    st.write(f"Top {num_recommendations} Recommended Jobs:")
    for i, row in recommendation.iterrows():
        st.write('- ' + row['Job_Name'] + ' at ' + row['Company'])
