import pandas as pd

import datetime

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_recall_fscore_support

from sklearn.preprocessing import LabelEncoder

# считываем файл с данными
df = pd.read_csv('casc-resto.csv', sep=';', decimal=',', dtype='str')

# приводим данные к "нормальному" виду
df['SummBasic'] = [x.replace(',', '.') for x in df['Sum']]
df['SummBasic'] = df['SummBasic'].astype(float)
df['SummAfterPointsUsage'] = df['SummAfterPointsUsage'].astype(float)
df['RKDate'] = pd.to_datetime(df['RKDate'], format='%Y-%m-%d')
df['SummAfterPointsUsage'] = [x.replace(',', '.') for x in df['SummAfterPointsUsage']]
df.drop_duplicates(keep=False, inplace=True)
df.dropna()

first_date = datetime.datetime(2017, 7, 1)
last_date = datetime.datetime(2017, 12, 31)

# сортировка по айди
df.sort_values("CustomerID", inplace=True)

# создаем датафреймы
before = df[df['RKDate'] < first_date]
after = df[df['RKDate'] >= first_date]

orders = before.copy()
orders.head()

# Recency
orders['dso'] = before['RKDate'].apply(lambda x: (first_date - x).days)
orders['dsf'] = before['RKDate'].apply(lambda x: (first_date - x).days)
aggr = {
    'dso': lambda x: x.min(),
    'dsf': lambda x: x.max(),
    'RKDate': lambda x: len([d for d in x]),
    'SummBasic': 'sum',
    'SummAfterPointsUsage': 'sum',
}

# доп.задание
df_1 = orders.groupby('CustomerID').agg(aggr).reset_index()
df_1['Dop'] = df_1['SummAfterPointsUsage'] - df_1['SummBasic']

# записать данные в файл
df_1.to_csv('df_2.csv', index=False, header=True)

# Recency, Frequency, Monetary Value
time = 30
df_2 = pd.read_csv('df_2.csv', sep=',', decimal='.')

df_2['SummBasic'].astype(float)
df_2['SummAfterPointsUsage'].astype(float)
df_2['visit'].astype(int)
df_2['Monetary'] = df_2['SummBasic'] / df_2['visit']
df_2['Frequency'] = df_2['visit'] / (
        df_2['DaysSinceFirst'] / time)
df['samedate'] = np.where((df['RKDate'] <= last_date) & (df['RKDate'] >= first_date), 1, 0)
df['samedate'] = df['samedate'].astype('int')
dfByDateMatch = df.groupby('CustomerID')['dateMatch'].agg(['sum'])

aggressive = {
    'sum': lambda x: len([d for d in x if d > 0])
}
dfY = dfByDateMatch.groupby('CustomerID').agg(aggressive)
dfY.rename(columns={'sum': 'Y', }, inplace=True)
dfY['Y'] = dfY['Y'].astype(int)


df2 = df_2[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Dop']].copy()
df2.set_index('CustomerID', inplace=True)

dfCustomers = pd.read_csv('CASC_Constant.csv', sep=',')
dfCustomers = dfCustomers.drop(columns=['ActivationDate', 'SubscribedEmail', 'SubscribedPush'])

dfCustomers.sort_values("CustomerId", inplace=True)
dfCustomers.rename(columns={"CustomerId": "CustomerID"}, inplace=True)
dfCustomers.set_index('CustomerID', inplace=True)

full = pd.merge(df2, dfCustomers, how='left', on=['CustomerID'])
full = pd.merge(full, dfY, how='left', on=['CustomerID'])

full['Age'] = full['Age'].astype(int)
full['Y'] = full['Y'].astype(int)
full = full[full.Monetary > 200]
full = full[full.Frequency < 20]
full = full.dropna()

label = LabelEncoder()
dicts = {}
label.fit(full.Sex.drop_duplicates())
dicts['Sex'] = list(label.classes_)
full.Sex = label.transform(full.Sex)
full = full.reset_index()
full = full.drop(columns=['CustomerID'])

y = full.Y
full = full.drop(columns=['Y'])
X = full

# Разбив данные проверяем успешность предсказаний тестовых и тренировочных данных
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(solver='liblinear')
model.fit(xtrain, ytrain)

y_pred = model.predict(xtest)

res_train = model.score(xtrain, ytrain)
res_test = model.score(xtest, ytest)

# !Вывод всех результатов!
print(df_1[['CustomerID', 'Recency', 'DaysSinceFirst', 'NumOFVisits']].head())
print(dfY)  # переменная y
print(df2)  # Задание 2
print(dfCustomers)  # По полу и возрасту
print(full)  # frequency, recency ...
print(res_train)  # успех на основе тренировочных данны
print(res_test)  # успех на основе тестовых данны
print(precision_recall_fscore_support(ytest, y_pred, average='macro'))  # Precision, recall based on Y
