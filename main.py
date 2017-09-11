import pandas as pd
import numpy as np
import graphlab
from sklearn.linear_model import LinearRegression

def makeCSV(df):
    # input = numpy array of
    # rows of [userid, jokeid, rating]
    filename = 'data/predictions.csv'
    df.to_csv(filename)

def avg_joke_score(df):
    jokes = df['joke_id'].unique()
    jokelist = []
    for i in xrange(151):
        if i in jokes:
            jokelist.append(i)

    Output = dict()
    for i in jokelist:
        Output[i] = 0.0
        Output[i] = df['rating'][df['joke_id']==i].mean()
    return Output

def get_data():
    ratings = pd.read_table("data/ratings.dat")
    ratings['joke_id'][ratings['joke_id']==151] = 15
    test_data = pd.read_csv("data/test_ratings.csv")
    avg_score = avg_joke_score(ratings)
    return ratings, test_data, avg_score

def recommender(ratings):
    sf = graphlab.SFrame(ratings)
    m1 = graphlab.factorization_recommender.create(sf, max_iterations=50, num_factors=2, linear_regularization=1e-12, user_id='user_id', item_id='joke_id', target='rating', solver='als')
    return m1

def testing_res(test_data, m1):
    test_sf = graphlab.SFrame(test_data)
    predicted_ratings = np.array(m1.predict(test_sf))
    output_df = test_data[['user_id','joke_id']]
    output_df['rating'] = predicted_ratings
    return output_df

def applying_linear_model(output_df, ratings, m1, sf):
    X_test = output_df.drop('user_id', axis=1)
    for i in avg_score:
        X_test['joke'+str(i)] = 0
        X_test['joke'+str(i)][X_test['joke_id']==i] = 1
    X_test = X_test.drop('joke_id', axis=1)
    X_test = np.array(X_test)

    test_lin_reg_df = ratings.drop('user_id', axis=1)
    for i in avg_score:
        test_lin_reg_df['joke'+str(i)] = 0
        test_lin_reg_df['joke'+str(i)][test_lin_reg_df['joke_id']==i] = 1
    test_lin_reg_df = test_lin_reg_df.drop('joke_id', axis=1)
    test_lin_reg_df = test_lin_reg_df.drop('rating', axis=1)

    test_prediction = np.array(m1.predict(sf))
    X = np.append(np.array([test_prediction]).T, np.array(test_lin_reg_df), axis=1)
    Y = np.array(ratings['rating'])
    LR = LinearRegression()
    LR.fit(X,Y)

    predicted = LR.predict(X_test)
    output_df['rating'] = predicted
    return output_df
