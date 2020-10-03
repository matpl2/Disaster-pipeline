import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load 2 dataframes from csv
    INPUT
    csv files loaded to 2 pandas data frame
    OUTPUT
    df - pandas DataFrame merged from 2 files
     
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    
    return df
    


def clean_data(df):
    """This function cleans data included in the DataFrame and transform categories
    INPUT
    df --  pandas DataFrame
    OUTPUT
    df -- cleaned pandas DataFrame
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.loc[0,:]
    

    
    category_colnames = row.apply(lambda x: x.split('-')[0]).tolist()
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    
    return df
    
    



def save_data(df, database_filename):
    """This function cleans data included in the DataFrame and transform categories
    INPUT
    df --  pandas DataFrame
    OUTPUT
    df -- clean
    """
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('messages_mp2', engine, index=False) 
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()