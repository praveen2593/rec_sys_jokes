# Recommendation System Jester Dataset

## Overview
Recommendation systems is one of the widely used ML concept. This project aims to develop my understanding of Recommendation systems using GraphLab and the Jester Dataset. Using the user ratings of the Jester Dataset, we built a recommendation system with Matrix Factorization. We then converted to an Ensemble model, by performing Linear Regression with manually tagged jokes. Error metric decreased significantly.

## Dataset
The dataset was downloaded from [Jester Dataset](http://eigentaste.berkeley.edu/dataset/)

## Data Cleaning and EDA
There were 150 jokes in all with ratings. 10 Jokes had zero rating and there was 1 rating without a joke id tag. We removed the jokes which had no rating, and reindexed the joke id.

## Files in src and it's use

* collection_app.py - Collects live data from a Heroku App and stores it into a MongoDB
* model.py - Compares the performance of models and stores the best model in pickle format
* my_app.py - Loads pickled model and performs the predictions on live data and pushes it to the Web app using Flask
* predict.py - Predicts the fraud risk level based on the probability

## Rough timeline 

* First 3 hours: EDA, Feature Engineering
* Next 3 hours: Model building and Deployment


## Credits
This project would not be possible without the efforts of my fellow teammaates Joseph Fang, Edward Rha, Elham Keshavarzian

