# 파일로부터 데이터 읽어오기
import pandas as pd
import tensorflow as tf
model = 0
레모네이드 = pd.read_csv(
    'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv')

보스턴 = pd.read_csv(
    'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv')

아이리스 = pd.read_csv(
    'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv')

print(레모네이드.shape, 레모네이드.columns)
print(보스턴.shape, 보스턴.columns)
print(아이리스.shape, 아이리스.columns)

독립 = 레모네이드[['온도']]
종속 = 레모네이드[['판매량']]
print(독립.shape, 종속.shape)


def model_structure(x, y):
    X = tf.keras.layers.Input(shape=[x])
    Y = tf.keras.layers.Dense(y)(X)
    global model
    model = tf.keras.models.Model(X, Y)
    model.compile(loss="mse")
    model.fit(독립, 종속, epochs=10000, verbose=1)
    model.fit(독립, 종속, epochs=10)


model_structure(1, 1)
print(model.predict(독립))
print(model.predict([[15]]))

독립 = 보스턴[['crim', 'zn', 'indus', 'chas', 'nox',
          'rm', 'age', 'dis', 'rad', 'tax',
          'ptratio', 'b', 'lstat']]
종속 = 보스턴[['medv']]
print(독립.shape, 종속.shape)

model_structure(13, 1)
print(model.predict(독립[5:10]))
print(종속[5:10])
print(model.get_weights())

독립 = 아이리스[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
종속 = 아이리스[['품종']]
print(독립.shape, 종속.shape)


print(레모네이드.head())
print(보스턴.head())
print(아이리스.head())
