from model import model

import numpy as np
import pandas as pd

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='ML Hyperparameter Tuning App', layout='wide')


# load data
df = pd.read_csv('data/heart.csv')

# streamlit app
st.write("""
# Machine Learning Hyperparameter Optimization App
### **(Heart Disease Claasification)**""")

# Displays the dataset
st.subheader('Dataset')
st.markdown('The **Heart Disease** dataset from kaggle is used.')
st.write(df.head(5))


# create sliders
# to create header in sidebar
st.sidebar.header('Set HyperParameters For Grid SearchCV')
split_size = st.sidebar.slider(
    'Data split ratio (% for Training Set)', 50, 90, 80, 5)

st.sidebar.subheader('Learning Parameters')
parameter_n_estimators = st.sidebar.slider(
    'Number of estimators for Random Forest (n_estimators)', 10, 500, (100, 200), 50)
parameter_n_estimators_step = st.sidebar.number_input(
    'Step size for n_estimators', 10, 50, 10, 10)

st.sidebar.write('---')
parameter_max_features = st.sidebar.multiselect(
    'Max Features (You can select multiple options)', ['auto', 'sqrt', 'log2'], ['auto'])

parameter_max_depth = st.sidebar.slider('Maximum depth', 5, 15, (5, 8), 2)
parameter_max_depth_step = st.sidebar.number_input(
    'Step size for max depth', 1, 3)

st.sidebar.write('---')
parameter_criterion = st.sidebar.selectbox('criterion', ('gini', 'entropy'))

st.sidebar.write('---')
parameter_cross_validation = st.sidebar.slider(
    'Number of Cross validation split', 2, 10, 5, 1)

st.sidebar.subheader('Other Parameters')
parameter_random_state = st.sidebar.number_input(
    'Enter Seed number (random_state)', 0, 1000, 42)
parameter_bootstrap = st.sidebar.selectbox('Bootstrap Samples', [True, False])
parameter_n_jobs = st.sidebar.selectbox(
    'Number of jobs to run in parallel (n_jobs)', options=[1, -1])

n_estimators_range = np.arange(
    parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)

max_depth_range = np.arange(
    parameter_max_depth[0], parameter_max_depth[1]+parameter_max_depth_step, parameter_max_depth_step)

param_grid = dict(max_features=parameter_max_features,
                  n_estimators=n_estimators_range, max_depth=max_depth_range)


if st.button('Build Model'):
    # some preprocessing steps
    print('Button pressed...')
    dataset = pd.get_dummies(
        df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    # st.write(dataset['target'].value_counts())
    split_size = 1 - split_size/100
    model(dataset, split_size, parameter_random_state,
          parameter_bootstrap, parameter_n_jobs, param_grid, parameter_cross_validation)
