# import libraries
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
nltk.download(['punkt','stopwords','wordnet'])
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
import sys


"""
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
"""


def main():

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)

        print('creating train and test sets...')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=2, test_size=0.25)

        print('building model...')
        model = build_model()

        print('training model...')
        model.fit(X_train, Y_train)

        print('evaluating model...')
        evaluate_model(Y, X_test, Y_test, model)

        print('saving model...\n    MODEL: {}')
        save_model(model, model_filepath)

        print('model saved to {}!'.format(model_filepath))

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'models/train_classifier.py data/DisasterResponse.db models/classifier.pkl')


def load_data(database_filepath):
    """
    load data from database
    :param database_path:
    :return:
    """

    df = pd.read_sql_table('messages', 'sqlite:///{}'.format(database_filepath))
    X = df['message']
    non_y_cols = ['id', 'message', 'original', 'genre']
    y_cols = [col for col in df.columns if col not in non_y_cols]
    Y = df[y_cols]
    return X, Y


def tokenize(text):
    """
    Tokenization function to process your text data
    :param text:
    :return:
    """

    # remove punctuation - optional
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Tokenize. Split sentence into sequence of words/tokens
    tokens = word_tokenize(text)

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize, lower, and strip whitespace
    # Create sequence of lemmatized tokens
    clean_tokens = []
    for t in tokens:
        clean_token = lemmatizer.lemmatize(t).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens


def build_model():
    """
     Build a machine learning pipeline
     tested different parameters w/gridsearch.
     Not running GridSearchCV to save processing time
    :return: MultiOutputClassifier model. result of last task in pipeline
    """


    # CountVectorizer - create word counts to tokenize, remove stopwords
    # TfidfTransformer - fit and transform word counts
    # MultiOutputClassifier - predictor

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, stop_words='english',
                                 ngram_range=(1,2))),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])

    # grid search below produced pipeline params above
    # parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
    #               'clf__estimator__max_depth': [None, 20, 30],
    #               'clf__estimator__min_samples_leaf': [1, 5, 10]}
    #
    # cv = GridSearchCV(pipeline, param_grid=parameters)
    # return cv

    return pipeline


def evaluate_model(Y, X_test, Y_test, model):
    """
    test model. print precision, recall, and f1 score for each Y category
    :param Y: Y df used to get column names
    :param X_test: X_test series
    :param Y_test: Y_test df
    :param model:
    :return:
    """

    Y_pred = model.predict(X_test)

    for i, c in enumerate(Y.columns):
        print("Category: ", c.upper())
        print(classification_report(Y_test[c].values, Y_pred[:, i]))

    eval_df = pd.DataFrame()

    for i, c in enumerate(Y.columns):
        eval_results = classification_report(Y_test[c].values, Y_pred[:, i], output_dict=True)
        eval_results = pd.DataFrame(eval_results).T
        eval_results.drop(['micro avg', 'macro avg', 'weighted avg'], inplace=True)
        eval_results['category'] = c
        eval_results['value'] = eval_results.index
        eval_results = eval_results[['category','value','precision','recall','f1-score','support']]
        eval_df = pd.concat([eval_df, eval_results], axis=0, ignore_index=True)

    eval_df.to_excel('models/classification_report.xlsx', index=False)
    print('classification report saved to models/classification_report.xlsx')


def save_model(model, model_filepath):
    """
    save model as pickle file
    :param model: final model
    :param dir: directory to save model
    :param model_filepath: model name
    :return:
    """

    pickle.dump(model, open('{}'.format(model_filepath), 'wb'))


if __name__ == "__main__":
    main()





