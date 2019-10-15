# Paul Turner


import sqlite3
import numpy
import pandas
import json
import math

import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def database_setup(training_data_json):
    categories = set()
    foodies = sqlite3.connect('foodies.db')
    foodies_cursor = foodies.cursor()
    foodies_cursor.execute('''
        CREATE TABLE IF NOT EXISTS food(
        id integer NOT NULL,
        category TEXT NOT NULL,
        UNIQUE(id));''')
    foodies_cursor.execute('''
        CREATE TABLE IF NOT EXISTS ingredients(
        id integer PRIMARY KEY AUTOINCREMENT,
        ingredient TEXT NOT NULL,
        UNIQUE(ingredient));''')
    foodies_cursor.execute('''
        CREATE TABLE IF NOT EXISTS bridge(
        food_id integer NOT NULL REFERENCES food(id),
        ingredients_id integer NOT NULL REFERENCES ingredients(id));''')
    foodies.commit()
    for food in training_data_json:
        categories.add(food['cuisine'])
        foodies_cursor.execute('''
            INSERT OR IGNORE INTO food(id, category)
            VALUES(?,?)''', [food['id'], food['cuisine']])
        item_id = foodies_cursor.lastrowid
        #print("\n")
        #print("FOOD ID")
        #print(item_id)
        for ingredient in food['ingredients']:
            foodies_cursor.execute('''
                SELECT id
                FROM ingredients
                WHERE ingredient = ?''', [ingredient])
            ingredient_id_tuple = foodies_cursor.fetchone()
            if ingredient_id_tuple is None:
                foodies_cursor.execute('''
                    INSERT INTO ingredients(id, ingredient)
                    VALUES(?,?);''', [None, ingredient])
                ingredient_id = foodies_cursor.lastrowid
            else:
                ingredient_id = ingredient_id_tuple[0]
            foodies_cursor.execute('''
                INSERT INTO bridge(food_id, ingredients_id)
                VALUES(?,?);''', [item_id, ingredient_id])
            #print(f'{ingredient_id} {ingredient}')
    foodies.commit()
    return categories


def tfidf():
    foodies = sqlite3.connect('foodies.db')
    foodies_cursor = foodies.cursor()
    foodies_cursor.execute('''
        CREATE TABLE IF NOT EXISTS tfidf(
        df integer,
        idf real,
        ingredient_id integer,
        FOREIGN KEY (ingredient_id) REFERENCES ingredient(id))''')
    foodies.commit()

    # since each ingredient will appear in each food only one time
    # the term frequency of each ingredient will be 1
    tf = 1

    foodies_cursor.execute('''
        SELECT COUNT(*)
        FROM ingredients;''')
    number_ingredients = foodies_cursor.fetchone()[0]

    foodies_cursor.execute('''
        SELECT COUNT(*)
        FROM food;''')
    number_foods = foodies_cursor.fetchone()[0]

    for ingredient in range(1, number_ingredients):
        foodies_cursor.execute('''
                SELECT COUNT(ingredients_id)
                FROM bridge
                WHERE ingredients_id = ?;''', [ingredient])
        df = foodies_cursor.fetchone()[0]
        idf = math.log((number_foods / df), 2) + 1
        # print(f"\ndf {df}\nidf {idf}")
        # Only insert the ingredients that appear more then four times
        # and have an inverse document frequency greater then 1.5
        foodies_cursor.execute('''
            INSERT INTO tfidf(df, idf, ingredient_id)
            VALUES(?,?,?);''', [df, idf, ingredient])
    foodies.commit()
    foodies.close()


def full_setup():
    with open('trainfile2.json', 'r') as json_file:
        training_data_no_format = json.load(json_file)
    categories = database_setup(training_data_no_format)
    tfidf()
    print(f'The number of categories is {len(categories)}.')
    print(f'{categories}')


def faster_cooking(df=5, idf=1.2):
    foodies = sqlite3.connect('foodies.db')
    foodies_cursor = foodies.cursor()

    foodies_cursor.execute('''
        SELECT t.ingredient_id, i.ingredient
        FROM tfidf t
        INNER JOIN ingredients i ON t.ingredient_id == i.id
        WHERE t.df > ? AND idf > ?;''', [df, idf])
    ingredients_id = foodies_cursor.fetchall()

    col_names = []
    used_ingredients_id = []
    for ingredient_id in ingredients_id:
        ingredient = ingredient_id[1]
        col_names.append(ingredient)
        used_ingredients_id.append(ingredient_id[0])
    used_ingredients_id = set(used_ingredients_id)

    foodies_cursor.execute('''
        SELECT id, category 
        FROM food;''')
    foods_id = foodies_cursor.fetchall()

    foodies_cursor.execute('''
        SELECT *
        FROM bridge''')
    bridge = foodies_cursor.fetchall()

    foodies_cursor.execute('''
    SELECT COUNT(*)
    FROM ingredients''')
    ingredient_count = foodies_cursor.fetchone()

    data = numpy.zeros((len(foods_id), ingredient_count[0]))

    for bridge_i in bridge:
        if bridge_i[1] in used_ingredients_id:
            data[bridge_i[0] - 1, bridge_i[1]] = 1

    col_zero = numpy.argwhere(numpy.all(data[..., :] == 0, axis=0))
    data = numpy.delete(data, col_zero, axis=1)

    df = pandas.DataFrame(data)
    row_category = []
    for food in foods_id:
        row_category.append(food[1])
    df['category'] = row_category

    return df


def cross_val_scores(data_x, data_y):
    random_forest = RandomForestClassifier(n_estimators=100)
    support_vector = SVC()
    gaussian_bayes = GaussianNB()

    scores_per = []
    for clf, name in [(random_forest, 'Random Forest'),
                      (support_vector, 'Support Vector Machine'),
                      (gaussian_bayes, 'Bayes')]:
        print(name)
        scores = cross_val_score(clf,
                                 data_x,
                                 data_y,
                                 cv=5)
        print(scores)
        print(numpy.mean(scores))
        scores_per.append(scores)
    return scores


def main():
    parser = argparse.ArgumentParser(description='Used for initial setup')
    parser.add_argument('-s', '--setup', action='store_true')
    parser.add_argument('-cvall', '--crossvalall', action='store_true')
    args = parser.parse_args()

    if args.setup:
        full_setup()

    range_df = 15
    range_idf = 20

    acc_matrix = numpy.zeros((range_df, range_idf))

    for df in range(1, range_df):
        for k in range(1, range_idf):
            idf = (k / 10.0) + 1.0

            data = faster_cooking(df=df, idf=idf)
            data_y = numpy.ravel(data.loc[:, data.columns == 'category'])
            data_x = data.drop('category', axis=1)

            if args.crossvalall:
                cross_val_scores(data_x, data_y)

            train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=.33)

            rfc = RandomForestClassifier(n_estimators=100)

            rfc.fit(train_x, train_y)

            predicted_test = rfc.predict(test_x)

            acc_score = accuracy_score(test_y, predicted_test)

            print(acc_score)
            acc_matrix[df][k] = acc_score


if __name__ == '__main__':
    main()
