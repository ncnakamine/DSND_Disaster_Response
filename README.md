# DSND_Disaster_Response

## Summary
- This project creates an ETL pipepline and machine learning pipeline to clean data and predict disaster response categories based on received messages. Natural language processing is used to generate features. A MultiOutputClassifier is used to classify messages into categories.


## Prerequisites 
- Anaconda 3 https://www.anaconda.com/distribution/


## Files
```
data
- DisasterResponse.db                output database from process_data.py
- categories.csv                     input data
- messages.csv                       input data
- process_data.py                    script to clean input data

figures
- Category_Counts.png                horizontal barplot of category counts
- Genre_Counts.png                   barplot of genres
- Predicted_Actual_Counts.png        horizontal barplot of predicted and actual counts
- create_visuals.py                  script to create pngs in figure folder

models
- classification_report.xlsx         classification report for final model
- classifier.pkl.zip                 trained model as a pickle file. zipped to reduce size
- train_classifier.py                script to train classification model 

```


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans and stores data in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains and saves classifier
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    - (If desired) To run create_visuals.py to create pngs of relevant distributions
    	`python figures/create_visuals.py models/classifier.pkl data/DisasterResponse.db`

2. Run the following command in the app's directory to run your web app.
    `python run.py`


## Citations 
- [Udacity Data Science Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
- Data from [Figure Eight](https://www.figure-eight.com/)





