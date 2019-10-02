import sqlite3
import json


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


def main():
    with open('trainfile2.json', 'r') as json_file:
        training_data_no_format = json.load(json_file)
    categories = database_setup(training_data_no_format)
    print(f'The number of categories is {len(categories)}.')
    print(f'{categories}')


if __name__ == '__main__':
    main()






