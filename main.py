import telebot
import tensorflow as tf
import requests
import pickle
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
bot = telebot.TeleBot("6727812890:AAFIYjIm-HBntAI9q1POQ6KV1gXLyy6GzHM")


with open('settings.pkl', 'rb') as file:
    loaded_lists = pickle.load(file)

# Восстановление списка в переменные
settings_in, settings_out, names = loaded_lists

#sett = [[[0/1, 0/num], [0/1, 0/num] ....], ....]

now = {}
steps = {}


# 0 - start; use( 1 - name; 2 - .csv/messages --> .csv/message)

@bot.message_handler(commands=['start'])
def mainn(message):
    global steps
    bot.send_message(message.chat.id, '''
Для чего бот:
-Что-бы создать модель по предсказанию строк или чисел на основе каких-либо признаков (строк или чисел) (признаки могут быть разные и разного количества)
Это инструкция как пользоваться ботом:
-Сначала в настроите нейросеть с помощю комманды /create (когда вы напишите эту комманду и вам будет объяснено что делать дальше)
-Если вы знаете нейросеть, уже загруженную сюда или вы её сами только что загрузили, то можете получить предсказания для новых данных с помощю /use''')
    steps[message.chat.id] = 0

@bot.message_handler(commands=['create'])
def mmain(message):
    global steps
    global names
    bot.send_message(message.chat.id, 'Напишите название нейронной сети, которую будете испрользовать')

    steps[message.chat.id] = -10


@bot.message_handler(commands=['use'])
def main(message):
    global steps
    bot.send_message(message.chat.id, 'Напишите название нейронной сети, которую хотите использовать')
    steps[message.chat.id] = 1

@bot.message_handler()
def repeat(message):
    global names
    global steps
    global settings_in
    global settings_out
    global now

    if steps[message.chat.id] == 1:
        if message.text in names:
            now[message.chat.id] = names.index(message.text)
            bot.send_message(message.chat.id, 'Теперь отправте .csv файл таблицы со всеми признаками для получения ответов')
            steps[message.chat.id] = 2
        else:
            bot.send_message(message.chat.id, 'Такой нейросети нет в базе данных')





    if steps[message.chat.id] == -10:
        print(names)
        if message.text in names:
            bot.send_message(message.chat.id, 'Это название уже занято')
        else:
            names += [message.text]
            now[message.chat.id] = names.index(message.text)
            steps[message.chat.id] = -12
            bot.send_message(message.chat.id, 'Теперь вы должны перечислить информацию, с помощью которой нейросеть предсказывает ответ')
            bot.send_message(message.chat.id, 'Первый признак это число или строка? (напишите "число" или "строка")')
            settings_in += [[]]


    elif steps[message.chat.id] == -12:

        if message.text == 'строка':
            bot.send_message(message.chat.id, 'Какая максимальное количество символов у строки (все строки в которых больше символов будут урезаны при обучении нейронной сети)')

            settings_in[now[message.chat.id]] += [0, 0] # 0 - words, 1 - numb
            print(settings_in)
            steps[message.chat.id] = -12.5

        elif message.text == 'число':
            bot.send_message(message.chat.id, 'Следующий признак это число или строка или это все? (напишите "число", "строка" или "все")')
            settings_in[now[message.chat.id]] += [1, 0]

        elif message.text == 'все':
            print(settings_in)

            bot.send_message(message.chat.id, 'Теперь вы должны перечислить информацию, которую будет предсказывать модель')
            bot.send_message(message.chat.id, 'Первый признак это число или строка? (напишите "число" или "строка")')
            settings_out += [[]]

            steps[message.chat.id] = -13

        else:
            bot.send_message(message.chat.id,'напишите "число", "строка" или "все"')

    elif steps[message.chat.id] == -12.5:
        try:
            settings_in[now[message.chat.id]][-1] = int(message.text)
            bot.send_message(message.chat.id,
                             'Следующий признак это число или строка или это все? (напишите "число", "строка" или "все")')
            steps[message.chat.id] = -12

        except ValueError:
            bot.send_message(message.chat.id, 'Напишите число')

    elif steps[message.chat.id] == -13:

        if message.text == 'строка':
            bot.send_message(message.chat.id, 'Какая максимальное количество символов у строки (все строки в которых больше символов будут урезаны при обучении нейронной сети)')

            settings_out[now[message.chat.id]] += [0, 0]  # 0 - words, 1 - numb
            print(settings_out)
            steps[message.chat.id] = -13.5

        elif message.text == 'число':
            bot.send_message(message.chat.id,
                             'Следующий признак это число или строка или это все? (напишите "число", "строка" или "все")')
            settings_out[now[message.chat.id]] += [1, 0]

        elif message.text == 'все':
            print(settings_out)

            bot.send_message(message.chat.id, '''теперь сдлелайте .csv файл в таком формате:
            Первая строка - заголовки, они идут в том же порядке, в котором вы их назвали. С начала идут данные, на основе которых предсказывается ответ, потом данные самих ответов.
            Все остальные строки - сами данные
            ''')

            steps[message.chat.id] = -14

        else:
            bot.send_message(message.chat.id, 'напишите "число", "строка" или "все"')

    elif steps[message.chat.id] == -13.5:
        try:
            settings_out[now[message.chat.id]][-1] = int(message.text)
            bot.send_message(message.chat.id,
                             'Следующий признак это число или строка или это все? (напишите "число", "строка" или "все")')
            steps[message.chat.id] = -13

        except ValueError:
            bot.send_message(message.chat.id,
                             'Напишите число')


@bot.message_handler(content_types=['document'])
def handle_document(message):
    if steps[message.chat.id] == 2:
        if message.document.mime_type == 'text/csv':
            try:
                file_info = bot.get_file(message.document.file_id)
                downloaded_file = bot.download_file(file_info.file_path)

                with open(names[now[message.chat.id]]+'.csv', 'wb') as new_file:
                    new_file.write(downloaded_file)

                bot.reply_to(message, "Файл обрабатывается")

                df = pd.read_csv(names[now[message.chat.id]]+'.csv')
                for col in df.columns:
                    pct_missing = np.mean(df[col].isnull())
                    print('{} - {}%'.format(col, round(pct_missing * 100)))

                for col in df.columns:
                    if 'Unnamed:' in col:
                        df = df.drop(col, axis=1)

                k = 0
                for col in df.columns:
                    print(settings_in[now[message.chat.id]][k * 2], k)
                    if settings_in[now[message.chat.id]][k * 2] == 1:
                        df[col] = df[col].fillna(-999)
                    else:
                        df[col] = df[col].fillna('n')

                    k += 1

                for col in df.columns:
                    pct_missing = np.mean(df[col].isnull())
                    print('{} - {}%'.format(col, round(pct_missing * 100)))

                with open(f'{names[now[message.chat.id]]}.pickle', 'rb') as handle:
                    tokenizer = pickle.load(handle)

                inp = []
                for i in range(len(df[col])):
                    inp += [[]]
                    k = 0
                    for col in df.columns:
                        # print(k * 2 - len(settings_in[0]))
                        if settings_in[now[message.chat.id]][k * 2] == 1:
                            inp[i] += [df[col][i]]
                        else:
                            inp[i] += tokenizer.texts_to_matrix(str(df[col][i])[:settings_in[now[message.chat.id]][k * 2 + 1] + 1] + ' ' * (
                                        settings_in[now[message.chat.id]][k * 2 + 1] - len(str(df[col][i])))).tolist()[0]

                        k += 1

                inp = np.array(inp).astype(np.float32)
                # С данными покончено!!!!
                model = tf.keras.models.load_model(f"{names[now[message.chat.id]]}.h5")
                pred = model.predict(inp)
                print(len(pred[0]))

                dff = pd.DataFrame()

                k = 0
                for i in settings_out[now[message.chat.id]]:
                    try:
                        e = settings_out[now[message.chat.id]][k*2+1]
                    except IndexError:
                        break
                    print(k)
                    if k % 2 == 0:
                        if i == 1:
                            print(pred[: (k//2 + 1) * len(df[col])])
                            dff[f'{k} - ответ'] = pred[: (k//2 + 1) * len(df[col])].tolist()
                            pred = pred[: (k//2 + 1) * len(df[col])]
                        else:
                            #print(x,len(x))
                            p = ['']
                            m = 0
                            #print(len(u))
                            for x in pred:
                                print(x)
                                for j in range(1, len(x)//100 + 1):
                                    print(np.array([x[:j * 100][0]]), 'sgrsgr')
                                    print(tokenizer.index_word[np.array(x[:j * 100]).astype(np.float32).argmax(axis=1)[0]])
                                    p[m] += tokenizer.index_word[np.array(x[:j * 100]).astype(np.float32).argmax(axis=1)[0]]
                                p += ['']
                                m += 1

                            dff[f'{k} - ответ'] = np.array(p)


                    k += 1
                #print(p,'fgh')

                #dff = pd.DataFrame(pred, columns=['Ответы'])

                dff.to_csv('Ответы.csv', index=False)

                with open('Ответы.csv', 'rb') as file:
                    bot.send_document(message.chat.id, file)

                bot.send_message(message.chat.id, 'Результаты предсказаний в этом файле')
                print(pred)
            except IndexError:
                bot.send_message(message.chat.id,
                                 'Неправильные данные в .csv')


        else:
            bot.reply_to(message, "Пожалуйста, отправьте .csv файл")



















    if steps[message.chat.id] == -14:
        if message.document.mime_type == 'text/csv':
            file_info = bot.get_file(message.document.file_id)
            downloaded_file = bot.download_file(file_info.file_path)

            with open(names[now[message.chat.id]]+'.csv', 'wb') as new_file:
                new_file.write(downloaded_file)

            bot.reply_to(message, "Файл обрабатывается")

            df = pd.read_csv(names[now[message.chat.id]]+'.csv')
            for col in df.columns:
                pct_missing = np.mean(df[col].isnull())
                print('{} - {}%'.format(col, round(pct_missing * 100)))
            for col in df.columns:
                if 'Unnamed:' in col:
                    df = df.drop(col, axis=1)
            k = 0
            texts = ''
            for col in df.columns:
                pct_missing = np.mean(df[col].isnull())
                a = round(pct_missing * 100)

                print(a, col)
                if len(settings_in[now[message.chat.id]]) < k * 2 + 1:
                    print(col)
                    break
                if a > 60:
                    bot.send_message(message.chat.id, f'Столбец {col} будет удалён, так как в нём слишком много пропусков, при использовании нейросети удаляйте этот столбец')
                    print(settings_in[now[message.chat.id]], k, 'sd')
                    settings_in[now[message.chat.id]] = settings_in[now[message.chat.id]][:k * 2] + settings_in[now[message.chat.id]][k * 2 + 2:]
                    print(settings_in[now[message.chat.id]], 's')

                    df = df.drop(col, axis=1)
                elif a != 0:
                    print(settings_in[now[message.chat.id]][k * 2], 'f')
                    if settings_in[now[message.chat.id]][k * 2] == 1:
                        df[col] = df[col].fillna(-999)
                    else:
                        df[col] = df[col].fillna('n')

                k += 1
            k = 0
            for col in df.columns:
                try:
                    # print(df[col][0], settings_in[0][k * 2], col)
                    if settings_in[now[message.chat.id]][k * 2] == 0:
                        for i in df[col]:
                            texts += str(i)
                            print(i)
                except IndexError:
                    break
                k += 1
                print(texts)
            for col in df.columns:
                pct_missing = np.mean(df[col].isnull())
                print('{} - {}%'.format(col, round(pct_missing * 100)))

            tokenizer = Tokenizer(num_words=100, char_level=True)
            tokenizer.fit_on_texts(texts + ' MB')
#            print(len(tokenizer.texts_to_matrix(df['Name'][0]).tolist()[1]))

            inp = []
            out = []
            for i in range(len(df[col])):
                inp += [[]]
                k = 0
                r = 1
                for col in df.columns:
                    print(col)
                    # print(k * 2 - len(settings_in[0]))
                    if len(settings_in[now[message.chat.id]]) < k * 2 + 1 and r:
                        r = 0
                        out += [[]]
                    if r and settings_in[now[message.chat.id]][k * 2] == 1:
                        print(df[col][i], col, i, k * 2 - len(settings_in[now[message.chat.id]]))
                        inp[i] += [df[col][i]]
                    elif r:
                        inp[i] += tokenizer.texts_to_matrix(str(df[col][i])[:settings_in[now[message.chat.id]][k * 2 + 1] + 1] + ' ' * (
                                    settings_in[now[message.chat.id]][k * 2 + 1] - len(str(df[col][i])))).tolist()[0]
                        print(str(df[col][i])[:settings_in[now[message.chat.id]][k * 2 + 1] + 1] + ' ' * (
                                    settings_in[now[message.chat.id]][k * 2 + 1] - len(str(df[col][i]))))
                    elif settings_out[now[message.chat.id]][k * 2 - len(settings_in[now[message.chat.id]])] == 1:
                        out[i] += [df[col][i]]
                    else:
                        print(df[col][i], col, i)
                        #print(tokenizer.texts_to_matrix(str(df[col][i]) + ' ' * (settings_out[now[message.chat.id]][k * 2 + 1] - len(str(df[col][i])))).tolist()[0])
                        out[i] += tokenizer.texts_to_matrix(str(df[col][i]) + ' ' * (settings_out[now[message.chat.id]][k * 2 - len(settings_in[now[message.chat.id]])] - len(str(df[col][i])))).tolist()[0]

                    k += 1

            inp = np.array(inp).astype(np.float32)
            out = np.array(out)
            print(inp)
            print(out)
            print(texts,'fds')

            # С данными покончено!!!!
            model = tf.keras.Sequential()
            model.add(Dense((len(inp[0]) + len(out[0])) // 2))
            model.add(Dense(len(out[0])))
            model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(1e-4))

            history = model.fit(x=inp, y=out, batch_size=32, epochs=1000)

            model.save(names[now[message.chat.id]]+".h5") # модель сохранена

            with open('settings.pkl', 'wb') as file:
                pickle.dump((settings_in, settings_out, names), file)

            with open(f'{names[now[message.chat.id]]}.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            bot.send_message(message.chat.id,
                             'Нейросеть обучена')


        else:
            bot.reply_to(message, "Пожалуйста, отправьте .csv файл")


bot.polling(none_stop = True)