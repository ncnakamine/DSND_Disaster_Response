# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


"""
python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db
"""


def main():

    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
            .format(messages_filepath, categories_filepath))
        df = load_and_merge_data(messages_filepath, categories_filepath)
        print('cleaning data...')
        df = clean_data(df)
        print('saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        print('cleaned data saved to {}!'.format(database_filepath)) 

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'data/disaster_messages.csv data/disaster_categories.csv '\
              'data/DisasterResponse.db')



def load_and_merge_data(messages_filepath, categories_filepath):
    """
    :param dir: directory where messages.csv and categories.csv live
    :return: merged dataframe of messages.csv and categories.csv
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df


def clean_data(df):
    """
    Split categories into separate category columns.
    Convert category values to just numbers 0 or 1.
    Replace categories column in df with new category columns.
    Remove duplicates.
    :param df: dataframe to clean
    :return: cleaned dataframe
    """

    # Split categories into separate category columns.
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[1]

    # use this row to extract a list of new column names for categories
    category_colnames = [i.split('-')[0] for i in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Replace categories column in df with new category columns.

    # drop the original categories column from df
    df.drop(labels=['categories'], axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner', ignore_index=False)

    # Remove duplicates
    df.drop_duplicates(keep='first', inplace=True)

    return df


def save_data(df, database_filepath):

    """
    save DisasterResponse.db
    :param df: dataframe to save
    :param dir: directory to save DisasterResponse.db to
    :return:
    """

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql(name='messages', con=engine, index=False)



if __name__ == "__main__":
    main()





