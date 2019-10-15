import sqlite3
import json
import math
import numpy
import pandas

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


def database_setup(training_data_json):
    categories = set()
    foodies = sqlite3.connect('foodies.db')
    foodies_cursor = foodies.cursor()
    foodies_cursor.execute('''
        CREATE TABLE IF NOT EXISTS food(
        id integer PRIMARY KEY AUTOINCREMENT,
        food_id integer NOT NULL,
        category TEXT NOT NULL,
        UNIQUE(food_id));''')
    foodies_cursor.execute('''
        CREATE TABLE IF NOT EXISTS ingredients(
        ingredient_id integer PRIMARY KEY AUTOINCREMENT,
        ingredient TEXT NOT NULL,
        UNIQUE(ingredient));''')
    foodies_cursor.execute('''
        CREATE TABLE IF NOT EXISTS bridge(
        food_id integer NOT NULL REFERENCES food(id),
        ingredients_id integer NOT NULL REFERENCES ingredients(ingredient_id));''')
    foodies.commit()
    for food in training_data_json:
        categories.add(food['cuisine'])
        foodies_cursor.execute('''
            INSERT OR IGNORE INTO food(food_id, category)
            VALUES(?,?)''', [food['id'], food['cuisine']])
        item_id = foodies_cursor.lastrowid
        #print("\n")
        #print("FOOD ID")
        #print(item_id)
        for ingredient in food['ingredients']:
            foodies_cursor.execute('''
                SELECT ingredient_id
                FROM ingredients
                WHERE ingredient = ?''', [ingredient])
            ingredient_id_tuple = foodies_cursor.fetchone()
            if ingredient_id_tuple is None:
                foodies_cursor.execute('''
                    INSERT INTO ingredients(ingredient_id, ingredient)
                    VALUES(?,?);''', [None, ingredient])
                ingredient_id = foodies_cursor.lastrowid
            else:
                ingredient_id = ingredient_id_tuple[0]
            foodies_cursor.execute('''
                INSERT INTO bridge(food_id, ingredients_id)
                VALUES(?,?);''', [item_id, ingredient_id])
            #print(f'{ingredient_id} {ingredient}')
    foodies.commit()
    foodies.close()
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
        if df > 4 and idf > 1.5:
            foodies_cursor.execute('''
                INSERT INTO tfidf(df, idf, ingredient_id)
                VALUES(?,?,?);''', [df, idf, ingredient])
    foodies.commit()
    foodies.close()


def faster_cooking():
    foodies = sqlite3.connect('foodies.db')
    foodies_cursor = foodies.cursor()

    foodies_cursor.execute('''
        SELECT t.ingredient_id, i.ingredient
        FROM tfidf t
        INNER JOIN ingredients i ON t.ingredient_id == i.ingredient_id;''')
    ingredients_id = foodies_cursor.fetchall()

    col_names = []
    used_ingredients_id = []
    for ingredient_id in ingredients_id:
        ingredient = ingredient_id[1]
        col_names.append(ingredient)
        used_ingredients_id.append(ingredient_id[0])
    used_ingredients_id = set(used_ingredients_id)

    foodies_cursor.execute('''
        SELECT food_id, category 
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


def main():
    with open('trainfile2.json', 'r') as json_file:
        training_data_no_format = json.load(json_file)
    categories = database_setup(training_data_no_format)
    tfidf()
    data = faster_cooking()

    random_forest = RandomForestClassifier(n_estimators=1000)
    support_vector = SVC()
    guassian_bayes = GaussianNB()

    scores_per = []
    for clf, name in [(random_forest, 'Random Forest'),
                        (support_vector, 'Support Vector Machine'),
                        (guassian_bayes, 'Bayes')]:
        print(name)
        scores = cross_val_score(clf,
                                 data.loc[:, data.columns != 'category'],
                                 data.loc[:, data.columns == 'category'],
                                 cv=None)
        print(scores)
        print(numpy.mean(scores))
        scores_per.append(scores)


if __name__ == '__main__':
    main()






