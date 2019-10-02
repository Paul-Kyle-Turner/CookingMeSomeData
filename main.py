import sqlite3
import math


def main():
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


if __name__ == '__main__':
    main()
