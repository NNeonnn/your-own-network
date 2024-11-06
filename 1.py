# settings.pkl = [[],[],[]]
import pickle

list1 = [[],[],[]]

# Сохранение списков в файл
with open('settings.pkl', 'wb') as file:
    pickle.dump((list1), file)