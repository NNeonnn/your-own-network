import telebot
import tensorflow as tf
import pickle
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
#tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'), 'GPU')
#print(f"TensorFlow version: {tf.__version__}")
#print(f"Cuda version: {tf.test.gpu_device_name()}")
bot = telebot.TeleBot("Your Telegram Bot Token")


with open('settings.pkl', 'rb') as file:
    loaded_lists = pickle.load(file)
# Восстановление списка в переменные
settings_in, settings_out, names = loaded_lists
print(settings_in, settings_out, names)

#sett = [[[0/1, 0/num], [0/1, 0/num] ....], ....]

now = {}
steps = {}


def maxlen(x):
    x = map(str, list(x))
    m = ''
    for i in x:
        if len(m) < len(i):
            m = i
    return len(m)







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

@bot.message_handler(commands=['settings'])
def mainn(message):
    global steps
    bot.send_message(message.chat.id, 'Напишите название неросети настройки которой вы хотите посмотреть')
    steps[message.chat.id] = 99

@bot.message_handler(commands=['create'])
def mmain(message):
    global steps
    global names
    bot.send_message(message.chat.id, 'Напишите название нейронной сети, которую хотите создать')

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
            steps[message.chat.id] = -14
            # bot.send_message(message.chat.id, 'Теперь вы должны перечислить информацию, с помощью которой нейросеть предсказывает ответ, а нет - хуй вам!!!!')
            bot.send_message(message.chat.id, '''теперь сдлелайте .csv файл в таком формате:
                        Первая строка - заголовки. С начала идут данные, на основе которых предсказывается ответ, потом пропуск одной строки и дальше данные самих ответов.
                        Все остальные строки - сами данные (в столбце где был пропуск пусто и дальше ответы)
                        ''')
            settings_in += [[]]


    if steps[message.chat.id] == 99:
        if message.text in names:
            now[message.chat.id] = names.index(message.text)
            a = settings_in[now[message.chat.id]]
            b = settings_out[now[message.chat.id]]
            a1 = ''
            c = 0
            k = 0
            for i in a:
                c = not c
                if c:
                    if i == 1:
                        a1 += 'число, '
                    else:
                        a1 += 'строка ('
                        k = 1
                elif k:
                    k = 0
                    a1 += f'{i}), '
            a1 = a1[:-2]

            b1 = ''
            c = 0
            k = 0
            for i in b:
                c = not c
                if c:
                    if i == 1:
                        b1 += 'число, '
                    else:
                        b1 += 'строка ('
                        k = 1
                elif k:
                    k = 0
                    b1 += f'{i}), '
            b1 = b1[:-2]

            bot.send_message(message.chat.id, f'''
Входные данные: {a1}
Выходные данные: {b1}
(В скобках после срока указана их максимальная длина)''')
            steps[message.chat.id] = 0
        else:
            bot.send_message(message.chat.id, 'Такой нейросети нет в базе данных')


@bot.message_handler(content_types=['document'])
def handle_document(message):
    global settings_in
    global settings_out
    if steps[message.chat.id] == 2:
        if message.document.mime_type == 'text/csv':
            try:
                file_info = bot.get_file(message.document.file_id)
                downloaded_file = bot.download_file(file_info.file_path)

                with open(names[now[message.chat.id]]+'.csv', 'wb') as new_file:
                    new_file.write(downloaded_file)

                bot.reply_to(message, "Файл обрабатывается")

                df = pd.read_csv(names[now[message.chat.id]]+'.csv')
                print(names[now[message.chat.id]]+'.csv')
                df = df.dropna(how='all')
                df.reset_index(drop=True, inplace=True)
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
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        df[col] = df[col].fillna('n')

                    k += 1

                for col in df.columns:
                    pct_missing = np.mean(df[col].isnull())
                    print('{} - {}%'.format(col, round(pct_missing * 100)))

                with open(f'{names[now[message.chat.id]]}.pickle', 'rb') as handle:
                    tokenizer = pickle.load(handle)
                _1 = tokenizer.texts_to_matrix('aa').tolist()
                print(_1, len(_1))
                inp = []
                for i in range(len(df)):
                    inp += [[]]
                    k = 0
                    for col in df.columns:
                        # print(k * 2 - len(settings_in[0]))
                        if settings_in[now[message.chat.id]][k * 2] == 1:
                            inp[i] += [df[col][i]]
                        else:
                            ss = tokenizer.texts_to_matrix(str(df[col][i])[:settings_in[now[message.chat.id]][k * 2 + 1] + 1] + ' ' * (
                                        settings_in[now[message.chat.id]][k * 2 + 1] - len(str(df[col][i])))).tolist()
                            sss = []
                            for jj in ss:
                                sss += jj
                            inp[i] += sss
                        k += 1

                inp = np.array(inp).astype(np.float32)
                print((inp))
                # С данными покончено!!!!
                model = tf.keras.models.load_model(f"{names[now[message.chat.id]]}.h5")
                pred = model.predict(inp)
                print(pred[0], len(pred[0]))

                dff = pd.DataFrame()

                k = 0
                #print(settings_out[now[message.chat.id]], "@@@@@", len(df), len(df[col]), col)
                for i in settings_out[now[message.chat.id]]:
                    try:
                        e = settings_out[now[message.chat.id]][k*2+1]
                    except IndexError:
                        break
                    print(k)
                    if k % 2 == 0:
                        if i == 1:
                            #print(pred[: (k//2 + 1) * len(df[col])])
                            dff[f'{k} - ответ'] = pred[: (k//2 + 1) * len(df)].tolist() # very strange!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            pred = pred[: (k//2 + 1) * len(df)]
                        else:
                            #print(x,len(x))
                            p = ['']
                            m = 0
                            #print(len(u))
                            for x in pred:
                                print(len(x), x, 'there')
                                for j in range(1, len(x)//100 + 1):
                                    #print(np.array([x[:j * 100][0]]), 'sgrsgr', x[:j * 100])
                                    #print(tokenizer.index_word[np.array(x[:j * 100]).astype(np.float32).argmax(axis=1)[0]])
                                    #print(np.array(x[(j - 1) * 100:j * 100]))
                                    index = int(tf.argmax(np.array(x[(j - 1) * 100:j * 100])))
                                    try:
                                        print(tokenizer.index_word[index])
                                        p[m] += tokenizer.index_word[index]
                                    except KeyError:
                                        break
                                p += ['']
                                m += 1

                            dff[f'{k} - ответ'] = np.array(p)


                    k += 1
                #print(p,'fgh')

                #dff = pd.DataFrame(pred, columns=['Ответы'])

                dff.to_csv(f'{str(message.chat.id)}Ответы.csv', index=False)

                with open(f'{str(message.chat.id)}Ответы.csv', 'rb') as file:
                    bot.send_document(message.chat.id, file)

                bot.send_message(message.chat.id, 'Результаты предсказаний в этом файле')
                print(pred)
            except IndexError and pd.errors.ParserError:
                bot.send_message(message.chat.id,
                                 'Неправильные данные в .csv')


        else:
            bot.reply_to(message, "Пожалуйста, отправьте .csv файл")



















    if steps[message.chat.id] == -14:
        if message.document.mime_type == 'text/csv':
            try:
                file_info = bot.get_file(message.document.file_id)
                downloaded_file = bot.download_file(file_info.file_path)

                with open(names[now[message.chat.id]]+'.csv', 'wb') as new_file:
                    new_file.write(downloaded_file)

                bot.reply_to(message, "Файл обрабатывается")

                df = pd.read_csv(names[now[message.chat.id]]+'.csv')
                print(df, "find this")
                df = df.dropna(how='all')
                df.reset_index(drop=True, inplace=True)
                print(df)
                for col in df.columns:
                    pct_missing = np.mean(df[col].isnull())
                    if str(pct_missing) != 'nan':
                        print('{} - {}%'.format(col, round(pct_missing * 100)))

                k = 0
                _ = 1
                for col in df.columns:
                    pct_missing = np.mean(df[col].isnull())
                    a = round(pct_missing * 100)

                    print(a, col)
                    #if len(settings_in[now[message.chat.id]]) < k * 2 + 1:
                    #    print(col)
                    #    break
                    if a == 100 and _:
                        _ = 0
                        continue
                    elif a > 60:
                        bot.send_message(message.chat.id, f'Столбец {col} будет удалён, так как в нём слишком много пропусков, при использовании нейросети удаляйте этот столбец')

                        df = df.drop(col, axis=1)
                    elif a != 0:
                        if str(df[col][1]).isdigit():
                            df[col] = df[col].fillna(df[col].mean())
                        else:
                            df[col] = df[col].fillna('n')

                    k += 1

                # v2 part
                settings_in += [[]]
                settings_out += [[]]
                _ = 0
                for col in df.columns:
                        print(col, 'how', col[:7])
                        if col[:7] == 'Unnamed':
                            if _:
                                print('break')
                                break
                            else:
                                print(1)
                                _ = 1
                                continue
                        elif _:
                            if not (pd.api.types.is_numeric_dtype(df[col])):
                                x1 = 0
                                x2 = maxlen(df[col])
                            else:
                                x1 = 1
                                x2 = 0
                            print('out', x1, x2)
                            settings_out[now[message.chat.id]] += [x1, x2]

                        else:
                            if not (pd.api.types.is_numeric_dtype(df[col])):
                                x1 = 0
                                x2 = maxlen(df[col])
                            else:
                                x1 = 1
                                x2 = 0
                            print('in', x1, x2)
                            settings_in[now[message.chat.id]] += [x1, x2]

                # df = df.drop(col, axis=1)
                if (len(settings_out[now[message.chat.id]]) == 0):
                    print(settings_out[now[message.chat.id]][10])
                print(settings_in[now[message.chat.id]], settings_out[now[message.chat.id]], 'yeah')
                # v2 part

                texts = set()
                k = 0
                _ = settings_in[now[message.chat.id]] + settings_out[now[message.chat.id]]
                for col in df.columns:
                    try:
                        # print(df[col][0], settings_in[0][k * 2], col)
                        if _[k * 2] == 0:
                            for i in df[col]:
                                texts = texts | set(str(i))
                                #print(i)
                    except IndexError:
                        break
                    k += 1
                    #print(texts)
                #for col in df.columns:
                    #pct_missing = np.mean(df[col].isnull())
                    #print('{} - {}%'.format(col, round(pct_missing * 100)))
                text = ''
                for i in texts:
                    text += i
                tokenizer = Tokenizer(num_words=len(text), char_level=True)
                tokenizer.fit_on_texts(text + ' MB')
                print(text,'fds')

    #            print(len(tokenizer.texts_to_matrix(df['Name'][0]).tolist()[1]))

                inp = []
                out = []
                print(settings_out[now[message.chat.id]])
                for i in range(len(df[col])):
                    #print(out, 'vot')
                    inp += [[]]
                    k = 0
                    r = 1
                    for col in df.columns:
                        if 'Unnamed:' in col:
                            continue
                        #print(k * 2, len(settings_in[now[message.chat.id]]), 'fff')
                        #print(col)
                        # print(k * 2 - len(settings_in[0]))
                        if len(settings_in[now[message.chat.id]]) < k * 2 + 1 and r:
                            r = 0
                            out += [[]]
                            k = 0
                        if r and settings_in[now[message.chat.id]][k * 2] == 1:
                            #print(df[col][i], col, i, k * 2 - len(settings_in[now[message.chat.id]]))
                            inp[i] += [df[col][i]]
                        elif r:
                            ss = tokenizer.texts_to_matrix(str(df[col][i])[:settings_in[now[message.chat.id]][k * 2 + 1] + 1] + ' ' * (
                                        settings_in[now[message.chat.id]][k * 2 + 1] - len(str(df[col][i])))).tolist()
                            #print(ss, 'wow')
                            sss = []
                            for j in ss:
                                sss += j
                            inp[i] += sss
                            #print(str(df[col][i])[:settings_in[now[message.chat.id]][k * 2 + 1] + 1] + ' ' * (
                            #            settings_in[now[message.chat.id]][k * 2 + 1] - len(str(df[col][i]))))

                        elif settings_out[now[message.chat.id]][k * 2] == 1:
                            out[i] += list([df[col][i]])
                            #print(list([df[col]]), 'ok')
                        else:
                            #print('hui')
                            #print(df[col][i], col, k, k * 2 + 1, settings_out[now[message.chat.id]])
                            #print(settings_out[now[message.chat.id]][k * 2 + 1] - len(str(df[col][i])), 'wtf')
                            ss = tokenizer.texts_to_matrix(str(df[col][i]) + ' ' * (settings_out[now[message.chat.id]][k * 2 + 1] - len(str(df[col][i])))).tolist()
                            #print(ss, 'wow')
                            sss = []
                            for j in ss:
                                sss += j
                            out[i] += sss
                        k += 1
                #test
                data = np.column_stack((inp, out))
                np.random.shuffle(data)
                inp = data[:, :-1]
                out = data[:, -1:]
                #test
                #m = 1/0
                inp = np.array(inp).astype(np.float64)
                out = np.array(out)
                print(inp, len(inp[0]))
                print(out, len(out[0]))
                #m = 1/0
                # С данными покончено!!!!
                model = tf.keras.Sequential()
                model.add(Dense((len(inp[0]) + len(out[0])) // 2 + 1, use_bias=True))
                model.add(Dense(len(out[0])))
                ma = -99999
                mi = 99999
                for i in out:
                    if max(i) > ma:
                        ma = max(i)
                    if min(i) < mi:
                        mi = min(i)
                stepp = (ma - mi) / 10000
                if stepp > 0.002:
                    stepp = 0.1
                if len(inp[0]) < 100:
                    ep = 500
                elif len(inp[0]) < 200:
                    ep = 200
                elif len(inp[0]) < 300:
                    ep = 100
                elif len(inp[0]) < 600:
                    ep = 50
                else:
                    ep = 20
                print(stepp)

                model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(stepp))
                _2 = 0
                lastM = 0
                for i in range(ep):
                    _2 += 1
                    if _2 / ep > 0.05:
                        if lastM != 0:
                            bot.delete_message(chat_id=message.chat.id, message_id=lastM)
                        _2 = 0
                        lastM = bot.send_message(message.chat.id, f'{round(i / ep * 100, 1)}%').message_id
                    history = model.fit(x=inp, y=out, batch_size=len(inp) // 32 + 1, epochs=1)
                    if sum(history.history['loss']) / len(history.history['loss']) < 0.06:
                        break
                bot.delete_message(chat_id=message.chat.id, message_id=lastM)

                model.save(names[now[message.chat.id]]+".h5") # модель сохранена

                with open('settings.pkl', 'wb') as file:
                    pickle.dump((settings_in, settings_out, names), file)

                with open(f'{names[now[message.chat.id]]}.pickle', 'wb') as handle:
                    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

                bot.send_message(message.chat.id,
                                 'Нейросеть обучена')
                steps[message.chat.id] = 0
            except IndexError:
                bot.send_message(message.chat.id,
                                 'Неправильные данные в .csv')


        else:
            bot.reply_to(message, "Пожалуйста, отправьте .csv файл")


bot.polling(none_stop = True)