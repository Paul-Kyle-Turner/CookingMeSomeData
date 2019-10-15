# Most of this file was taken from https://www.kaggle.com/alonalevy/cultural-diffusion-by-recipes
# Author alona_levy

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


# functionized version of alona_levy code
def gather_data(file='trainfile2.json'):
    with open(file) as data_file:
        data = json.load(data_file)
    return data


# ripped function from alona_levy
# some of the variable names have been changed to be more easily read
def create_dict_cuisine_ingredients(data_json):
    dictionary_cuisine_ingredient = {}
    cuisines = []
    ingredients = []

    for i in range(len(data_json)):

        # removed name change
        cuisine = data_json[i]['cuisine']

        ingredients_per_cuisine = data_json[i]['ingredients']

        if cuisine not in dictionary_cuisine_ingredient.keys():
            cuisines.append(cuisine)
            dictionary_cuisine_ingredient[cuisine] = ingredients_per_cuisine

        else:
            current_list = dictionary_cuisine_ingredient[cuisine]
            current_list.extend(ingredients_per_cuisine)
            dictionary_cuisine_ingredient[cuisine] = current_list

        ingredients.extend(ingredients_per_cuisine)

    ingredients = list(set(ingredients))  # unique list of ALL ingredients
    num_unique_ingredients = len(ingredients)
    num_cuisines = len(cuisines)

    return dictionary_cuisine_ingredient, num_cuisines, num_unique_ingredients, cuisines, ingredients


# ripped function from alona_levy
# variable names changed for better readability
# slight changes for the comments
def create_term_count_matrix(dictionary_cuisine_ingredient,
                             num_cuisines, num_unique_ingredients, cuisines, ingredients):
    term_count_matrix = np.zeros((num_cuisines, num_unique_ingredients))
    i = 0

    for cuisine in cuisines:
        ingredients_per_cuisine = dictionary_cuisine_ingredient[cuisine]

        for ingredient in ingredients_per_cuisine:
            j = ingredients.index(
                ingredient)
            # in order to know which column to put the term count in,
            # we will ago according to the terms' order in the ingredients array
            term_count_matrix[i, j] += 1

        i += 1

    return term_count_matrix


# ripped function from alona_levy
# variable names changed for better readability
def tf_idf_from_count_matrix(counts_matrix):
    counts_matrix = sparse.csr_matrix(counts_matrix)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(counts_matrix)  # normalizes vectors to mean 0 and std 1 and computes tf-idf
    tfidf.toarray()
    return tfidf.toarray()


def main():
    data = gather_data()

    data_frame = pd.read_json('trainfile2.json')
    data_frame['ingredients_clean_string'] = [' , '.join(z).strip() for z in data_frame['ingredients']]
    data_frame['ingredients_string'] = [
        ' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip().lower() for lists in
        data_frame['ingredients']]
    data_frame['ingredients_under_string'] = [
        ' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', '_', line)) for line in lists]).strip().lower() for lists in
        data_frame['ingredients']]

    corpus_train = data_frame['ingredients_string']
    vectorizer_total = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer_total.fit_transform(corpus_train).todense()

    x_train, x_test, y_train, y_test = train_test_split(data_frame.drop(['cuisine'], axis=1), data_frame['cuisine'])

    corpus_train = x_train['ingredients_string']
    tfidf_train = vectorizer_total.transform(corpus_train).todense()

    corpusts = x_test['ingredients_string']
    tfidf_test = vectorizer_total.transform(corpusts)

    # classifier = LinearSVC(C=0.80, penalty="l2", dual=False)
    #parameters = {'C': [1, 10]}
    # clf = LinearSVC()
    #clf = LogisticRegression()

    #classifier = GridSearchCV(clf, parameters)
    #print("something")
    #classifier = classifier.fit(tfidf_train, y_train)

    #dictionary_cuisine_ingredient, num_cuisines, \
        #num_unique_ingredients, set_cuisines, ingredients = create_dict_cuisine_ingredients(data)

    #term_count_matrix = create_term_count_matrix(dictionary_cuisine_ingredient, num_cuisines,
                                                 #num_unique_ingredients, set_cuisines, ingredients)

    # this will be the data matrix for the rest of the classifiers
    #tfidf_matrix = tf_idf_from_count_matrix(term_count_matrix)

    #data_frame = pd.DataFrame(tfidf_matrix)

    #cuisine = []
    #for food in data:
        #cuisine.append(food['cuisine'])

    #x_train, x_test, y_train, y_test = train_test_split(tfidf_matrix, cuisine)

    params_random_forest = {'n_estimators': [10, 100, 1000], 'criterion': ["gini", "entropy"]}
    random_forest = RandomForestClassifier(n_estimators=1000)
    params_support_vector = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'), 'C': [1]}
    support_vector = SVC()

    for classifier, name, params in [#[random_forest, 'Random_forest', params_random_forest],
                                     [support_vector, "Support_vector_machine", params_support_vector]]:
        grid_searched = GridSearchCV(classifier, params, cv=5, n_jobs=2)
        print(name)
        grid_searched.fit(tfidf, data_frame['cuisine'])
        pickle.dump(grid_searched, open(f'{name}' + str('.p'), 'wb'))


if __name__ == '__main__':
    main()
