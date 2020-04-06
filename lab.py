import pandas

import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

# Считываем файлы с данными
df_1 = pandas.read_csv('casc-resto.csv', sep=';', decimal=',', dtype='str')
df_2 = pandas.read_csv('CASC_Constant.csv', sep=',', decimal=',', dtype='str')
df_3 = pandas.DataFrame(columns=['CustomerID', 'Frequency', 'Recency', 'Monetary Value', 'Mean Saved', 'Visit'])

first_date = datetime.datetime(2017, 7, 1)
last_date = datetime.datetime(2017, 12, 31)

# Период в 30 дней
time = 30

# Приведение данных к "нормальному" виду
df_2.rename(columns={'CustomerId': 'CustomerID'}, inplace=True)

df_1['RKDate'] = pandas.to_datetime(df_1['RKDate'])

df_1['SumBasic'] = df_1['SumBasic'].astype(float) # Приводим значения к вещественному типу
df_1['SumBasic'] = [i.replace(',', '.') for i in df_1['SumBasic']] # Заамена запятых на точки

df_1['SumAfterPointsUsage'] = df_1['SumAfterPointsUsage'].astype(float) # Приводим значения к вещественному типу
df_1['SumAfterPointsUsage'] = [i.replace(',', '.') for i in df_1['SumAfterPointsUsage']] # Заамена запятых на точки

# Количество раз, когда SumAfterPointsUsage меньше чем SumBasic
use_freq = 0

# Подсчет use_freq
for item in df_1:
    if df_1['SumAfterPointsUsage'] < df_1['SumBasic']:
        use_freq += 1

for a, b in df_1.groupby(['CustomerID']):
    date = b[b['RKDate'] < first_date]

    if not date.empty:
        Frequency = date[0] / ((first_date - min(date['RKDate'])).days / time)

        Recency = first_date - max(date['RKDate'])

        MonetaryValue = date['SumBasic'].mean()

        MeanSaved = (date['SumBasic'] - date['SumAfterPointsUsage']).mean()

        df_1['Usage'] = date[0] / use_freq
        # Заполняем датафрейм полями и данными
        df_3 = df_3.append(
            {'CustomerID': a,
             'Frequency': Frequency,
             'Recency': Recency,
             'Monetary Value': MonetaryValue,
             'Mean Saved': MeanSaved,
             'Usage': df_1['Usage']}, ignore_index=True)


# объединение 2 и 3 датафрейма в один df_res
df_res = pandas.merge(df_3, df_2, on='CustomerID', how='left')

df_res['Age'] = df_res['Age'].astype(int)
df_res['Sex'] = df_res['Sex'].astype(int)
df_res['Usage'] = df_res['Usage'].astype(int)

# добавляем условия для исключения выбросов
if df_res[df_res.Monetary > 200]:
    df_res = df_res.dropna()
if df_res[df_res.Frequency < 20]:
    df_res = df_res.dropna()

y = df_res['Usage']
merged = df_res.drop(columns=['Usage'])

# Деление выборки на тестовую и тренировочную в соотношении 0.8 - 0.2
xtrain, xtest, ytrain, ytest = train_test_split(df_res, y, test_size=0.2)

regres = LogisticRegression()
regres.fit(xtrain, ytrain)

res_train = regres.score(xtrain, ytrain)
res_test = regres.score(xtest, ytest)

ypred = regres.predict(xtest)

# !Вывод всех результатов!
print(res_train)
print(res_test)
print(ytest, ypred)
print(precision_recall_fscore_support(ytest, ypred))
