import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib as plt
import sqlite3
import json


def main():
    with open('trainfile2.json', 'r') as json_file:
        training_data_no_format = json.load(json_file)
    categories = set()
    foodies = sqlite3.connect('foodies.db')
    foodies_cursor = foodies.cursor()
    foodies_cursor.execute('''
    CREATE TABLE IF NOT EXISTS food(
    id int IS NOT NONE,
    category TEXT IS NOT NONE);''')
    foodies_cursor.execute('''
        CREATE TABLE IF NOT EXISTS ingredients(
        ingredient TEXT IS NOT NONE,
        FOREIGN KEY (food_id) REFERENCES food(id));''')
    foodies.commit()
    for item in training_data_no_format:
        categories.add(item['cuisine'])
        foodies_cursor.execute('''
        INSERT INTO food(id, category)
        VALUES(?,?)
        WHERE NOT EXISTS(
        SELECT * FROM food
        WHERE id = ? AND category = ?)''', [item['id'], item['cuisine'], item['id'], item['cuisine']])
    print(f'The number of categories is {len(categories)}.')
    print(f'{categories}')


if __name__ == '__main__':
    main()






