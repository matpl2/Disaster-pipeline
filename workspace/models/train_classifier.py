import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
import warnings
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
warnings.simplefilter('ignore')



def load_data(database_filepath):
    """This function loads data from db and creates 2 data frames"""
    
    engine = create_engine('sqlite:///' + 'DisasterResponse.db')

    df = pd.read_sql_table('messages_mp2', con=engine) 

    X = df['message']
    df = df.drop(columns=['id','message','original','genre'])
    y = df[df.columns[4:]]
    category_names = y.columns
    return X, y, category_names
    


def tokenize(text):
     """
     This function tokenize data set
     """
     text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
     tokens = word_tokenize(text)
     lemmatizer = WordNetLemmatizer()
     clean_tokens = []
    
     for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

     return clean_tokens


def build_model():
    """Return Grid Search model with pipeline and Classifier"""
    moc = MultiOutputClassifier(RandomForestClassifier())
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', moc)
        ])

    parameters = {'clf__estimator__max_depth': [10, 50, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10]}

    cv = GridSearchCV(pipeline, parameters)
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """This function evaluates model"""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=category_names))
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    
    pass


def save_model(model, model_filepath):
    """Save model as pickle file"""
    pickle.dump(model, open(model_filepath, 'wb'))
    



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()