import plotly.graph_objects as go
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import plot_confusion_matrix

import streamlit as st


def model(dataset, split_size, parameter_random_state, parameter_bootstrap, parameter_n_jobs, param_grid, parameter_cross_validation):
    print('Model building...')
    Y = dataset['target']
    X = dataset.drop(['target'], axis=1)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=split_size, random_state=parameter_random_state)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # st.write(split_size, '% data for Training')

    rf = RandomForestClassifier(random_state=parameter_random_state,
                                bootstrap=parameter_bootstrap,
                                n_jobs=parameter_n_jobs)
    grid = GridSearchCV(estimator=rf, param_grid=param_grid,
                        cv=parameter_cross_validation)
    grid.fit(X_train, Y_train)
    # grid.best_estimator_

    st.subheader('Model Performance')
    Y_pred_test = grid.predict(X_test)
    st.write('Accuracy score of given model')
    st.write(accuracy_score(Y_test, Y_pred_test))
    st.write("The best parameters are %s with a score of %0.4f" %
             (grid.best_params_, grid.best_score_))
    # st.write(grid.get_params())

    # build 3D visualizer
    #-----Process grid data-----#
    grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]), pd.DataFrame(
        grid.cv_results_["mean_test_score"], columns=["accuracy"])], axis=1)

    grid_contour = grid_results.groupby(['max_depth', 'n_estimators']).mean()
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['max_depth', 'n_estimators', 'accuracy']
    grid_pivot = grid_reset.pivot('max_depth', 'n_estimators')
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values

    # define Layout and axis
    layout = go.Layout(
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='n_estimators')
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text='max_depth')
        ))
    fig = go.Figure(data=[go.Surface(z=z, y=y, x=x)], layout=layout)
    fig.update_layout(title='Hyperparameter tuning',
                      scene=dict(
                          xaxis_title='n_estimators',
                          yaxis_title='max_depth',
                          zaxis_title='accuracy'),
                      autosize=False,
                      width=800, height=800,
                      margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)

    st.subheader("Classification Report")

    # it will return output in the form of dictionary
    clf = classification_report(Y_test, Y_pred_test, labels=[
        0, 1], output_dict=True)
    st.write("""
    ### For Class 0(no disease) :
      Precision : %0.2f     
      Recall : %0.2f      
      F1-score  : %0.2f""" % (clf['0']['precision'], clf['0']['recall'], clf['0']['f1-score']))
    st.write("""
    ### For Class 1(has disease) :
      Precision : %0.3f    
      Recall : %0.3f      
      F1-score  : %0.3f""" % (clf['1']['precision'], clf['1']['recall'], clf['1']['f1-score']))

    st.subheader("Confusion Matrix")
    plot_confusion_matrix(grid, X_test, Y_test, display_labels=[
                          'No disease', 'Has disease'])

    st.pyplot()
