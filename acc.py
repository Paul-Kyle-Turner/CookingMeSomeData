import numpy as np
import pandas as pd
import json
import re
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pickle

if __name__ == '__main__':

    with open('Random_forest.p', 'rb') as file:
        random_forest = pickle.load(file)

    with open('Support_vector_machine.p', 'rb') as file:
        support_vector = pickle.load(file)

    data_frame = pd.read_pickle('train_frame.p')
    testdf = pd.read_json('testfile2a.json')

    testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
    testdf['ingredients_string'] = [
        ' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip().lower() for
        lists in
        testdf['ingredients']]
    testdf['ingredients_under_string'] = [
        ' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', '_', line)) for line in lists]).strip().lower() for
        lists in
        testdf['ingredients']]

    corpustr = data_frame['ingredients_string']
    vectorizertr = TfidfVectorizer(stop_words='english')
    tfidftr = vectorizertr.fit_transform(corpustr).todense()
    corpusts = testdf['ingredients_string']
    tfidfts = vectorizertr.transform(corpusts).todense()

    print(tfidfts)
    print(np.shape(tfidfts))

    best_predict = support_vector.predict(tfidfts)

    print(best_predict)
    print(len(best_predict))
    print(type(best_predict))

    results_data = {
        'id': testdf['id'],
        'cuisine': best_predict
    }

    print(np.shape(results_data))

    results_data_frame = pd.DataFrame(results_data)

    results_data_frame.to_csv('PaulTurner.csv', index=False)



