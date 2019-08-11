# DSND_Disaster_Response

## Summary
- This project creates an ETL pipepline and machine learning pipeline to clean data and predict disaster response categories based on received messages. Natural language processing was used to generate features. A MultiOutputClassifier was used to classify messages into categories.


## Prerequisites 
- Anaconda 3 https://www.anaconda.com/distribution/


## Files
```
data
- categories.csv                     input data
- DisasterResponse.db                output database from process_data.py
- messages.csv                       input data
- process_data.py                    script to clean input data

figures
- Category_Counts.png                horizontal barplot of category counts
- create_visuals.py                  script to create pngs in figure folder
- Genre_Counts.png                   barplot of genres
- Predicted_Actual_Counts.png        horizontal barplot of predicted and actual counts

models
- classification_report.xlsx         classification report for final model
- classifier.pkl                     trained model as a pickle file 
- train_classifier.py                script to train classification model 

```


## Citations 
- [Udacity Data Science Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
- Data from [Figure Eight](https://www.figure-eight.com/)





