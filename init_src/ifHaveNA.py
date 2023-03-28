###########################
# 라이브러리 사용
import pandas as pd

###########################
# 파일 읽어오기
파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris2.csv'
아이리스 = pd.read_csv(파일경로)
아이리스.head()

###########################
# 칼럼의 데이터 타입 체크
print(아이리스.dtypes)

# 원핫인코딩 되지 않는 현상 확인
인코딩 = pd.get_dummies(아이리스)
인코딩.head()

###########################
# 품종 타입을 범주형으로 바꾸어 준다.
아이리스['품종'] = 아이리스['품종'].astype('category')
print(아이리스.dtypes)

# 카테고리 타입의 변수만 원핫인코딩
인코딩 = pd.get_dummies(아이리스)
인코딩.head()

###########################
# NA값을 체크해 봅시다.
아이리스.isna().sum()
아이리스.tail()

###########################
# NA값에 꽃잎폭 평균값을 넣어주는 방법
mean = 아이리스['꽃잎폭'].mean()
print(mean)
아이리스['꽃잎폭'] = 아이리스['꽃잎폭'].fillna(mean)
아이리스.tail()

# Thanks to 이선비.
