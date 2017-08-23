# Recommendation System Jester Dataset
Using the user ratings of the Jester Dataset, built a recommendation system with Matrix Factorization. Converted to an Ensemble model, by performing Linear Regression with manually tagged jokes. Error metric decreased significantly. Performed unsupervised KMeans clustering to identify possible groups of joke within data. 

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

