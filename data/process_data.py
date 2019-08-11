
# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


"""
python process_data.py messages.csv categories.csv DisasterResponse
"""


def main():

    if len(sys.argv) == 4:
        messages_filename, categories_filename, database_name = sys.argv[1:]

    print('loading data...')
    df = load_and_merge_data(messages_filename, categories_filename)
    print('cleaning data...')
    df = clean_data(df)
    print('saving data...')
    save_data(df, database_name)
    print('data saved to {}'.format(sys.argv[3])) 


def load_and_merge_data(messages_filename, categories_filename):
    """
    :param dir: directory where messages.csv and categories.csv live
    :return: merged dataframe of messages.csv and categories.csv
    """

    messages = pd.read_csv(messages_filename)
    categories = pd.read_csv(categories_filename)
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


def save_data(df, database_name):

    """
    save DisasterResponse.db
    :param df: dataframe to save
    :param dir: directory to save DisasterResponse.db to
    :return:
    """

    engine = create_engine('sqlite:///{}.db'.format(database_name))
    df.to_sql(name='messages', con=engine, index=False)



if __name__ == "__main__":
    main()





