
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import re
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import sys 
from os.path import isdir,expanduser
import os


"""
python create_visuals.py classifier.pkl DisasterResponse.db 

"""


def main():

    if len(sys.argv) == 3:
        model_name, database_name = sys.argv[1:]

    model_path = os.path.split(os.getcwd())[0]+'/models/'
    model_name = model_path + model_name 
    database_path = os.path.split(os.getcwd())[0]+'/data/'
    database_name = database_path + database_name


    print('loading model...')
    model = load_model(model_name)

    print('loading data...')
    Y, Y_genre, X_test, Y_test, pred_actual_df = load_data(database_name, model)

    print('creating visuals...')
    create_visuals(Y, Y_genre, pred_actual_df)





def tokenize(text):
    """
    Tokenization function to process text data
    Required to load model
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



def load_model(model_name):
    """
    :param model_name:
    :return: model
    """

    pickle_off = open(model_name,"rb")
    model = pickle.load(pickle_off)
    return model


def load_data(database_name, model):
    """
    Load data necessary for visuals
    :param database_name:
    :param model:
    :return: Y, Y_genre, X_test, Y_test, pred_actual_df
    """

    df = pd.read_sql_table('messages', 'sqlite:///{}'.format(database_name))
    X = df['message']

    Y_genre = df[['genre']]

    non_y_cols = ['id', 'message', 'original', 'genre']
    y_cols = [col for col in df.columns if col not in non_y_cols]
    Y = df[y_cols]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=2, test_size=0.25)

    # create df for predicted vs actual counts
    Y_pred = model.predict(X_test)
    Y_pred_sums = [Y_pred[0:, i].sum() for i in range(len(Y.columns))]
    Y_test_sums = [Y_test[c].sum() for c in Y.columns]
    Y_dict = {'Categories': Y.columns, 'Predicted': Y_pred_sums, 'Actual': Y_test_sums}
    pred_actual_df = pd.DataFrame(Y_dict)
    pred_actual_df.index = pred_actual_df['Categories']
    pred_actual_df.drop('Categories', axis=1, inplace=True)


    return Y, Y_genre, X_test, Y_test, pred_actual_df






def create_visuals(Y, Y_genre, pred_actual_df):

    """
    create and save visuals
    :param Y:
    :param Y_genre:
    :param pred_actual_df:
    :return:
    """

    Y_counts = [Y[i].sum() for i in Y.columns]
    plt.barh(y=Y.columns, width=Y_counts, color = 'b')
    plt.xlabel('Counts')
    plt.title('Category Counts')
    plt.tight_layout()
    plt.savefig('Category_Counts.png')
    print('figure saved to Category_Counts.png')
    plt.clf()

    plt.bar(x=Y_genre['genre'].value_counts().index, height=Y_genre['genre'].value_counts(), color='b')
    plt.xlabel('Genre')
    plt.ylabel('Counts')
    plt.title('Genre Counts')
    plt.tight_layout()
    plt.savefig('Genre_Counts.png')
    print('figure saved to Genre_Counts.png')
    plt.clf()

    pred_actual_df.plot(kind='barh')
    plt.xlabel('Counts')
    plt.title('Predicted vs Actual Counts')
    plt.tight_layout()
    plt.savefig('Predicted_Actual_Counts.png')
    print('figure saved to Predicted_Actual_Counts.png')


if __name__ == "__main__":
    main()




