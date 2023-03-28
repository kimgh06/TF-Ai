##########################
# 라이브러리 사용
import tensorflow as tf
import pandas as pd

###########################
# 1.과거의 데이터를 준비합니다.
파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
보스턴 = pd.read_csv(파일경로)

# 종속변수, 독립변수
독립 = 보스턴[['crim', 'zn', 'indus', 'chas', 'nox',
          'rm', 'age', 'dis', 'rad', 'tax',
          'ptratio', 'b', 'lstat']]
종속 = 보스턴[['medv']]
print(독립.shape, 종속.shape)

###########################
# 2. 모델의 구조를 만듭니다
X = tf.keras.layers.Input(shape=[13])
H = tf.keras.layers.Dense(100000, activation='swish')(X)
Y = tf.keras.layers.Dense(1)(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

# 모델 구조 확인
model.summary()

###########################
# 3.데이터로 모델을 학습(FIT)합니다.
model.fit(독립, 종속, epochs=1000)

###########################
# 4. 모델을 이용합니다
print(model.predict(독립[:5]))
print(종속[:5])

###########################
# 모델의 수식 확인
print(model.get_weights())
