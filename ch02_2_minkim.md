# 기본 패키지


* 1.기본
```python
import numpy as np # numpy 패키지 가져오기(기본패키지)
import matplotlib.pyplot as plt # 시각화 패키지 가져오기
```

* 2.데이터 가져오기
```python
import pandas as pd # csv -> dataframe으로 전환
from sklearn import datasets # python 저장 데이터 가져오기
```

* 3.데이터 전처리
```python
from sklearn.preprocessing import StandardScaler # 연속변수의 범주화
from sklearn.preprocessing import LabelEncoder # 범주형 변수 수치화
```


* 4. 훈련/검증용 데이터 분리
```python
from sklearn.model_selection import train_test_split # 훈련용과 검증용 데이터 분리
```

* 5.분류모델구축
```python
from sklearn.tree import DecisionTreeClassifier # 결정트리
from sklearn.naive_bayes import GaussianNB # 나이브 베이즈
from sklearn.neighbors import KNeighborsClassifier # K-NN(K-최근접 이웃)
from sklearn.ensemble import RandomForestClassifier # 랜덤 포레스트
from sklearn.ensemble import BaggingClassifier # 앙상블
from sklearn.linear_model import Perceptron # 퍼셉트론
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀 모델
from sklearn.svm import SVC # 서포트 백터 머신(SVM)
from sklearn.neural_network import MLPClassifier #다층인공신경망
```

* 6.모델검정
```python
from sklearn.metrics import confusion_matrix, classification_report # 정오분류표
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer # 정확도, 민감도
from sklearn.metrics import roc_curve # ROC 곡선 그리기
```

* 7.최적화
```python
from sklearn.model_selection import cross_validate # 교차타당도
from sklearn.pipeline import make_pipeline # 파이프라인 구축
from sklearn.model_selection import learning_curve, validation_curve # 학습곡선,
from sklearn.model_selection import GridSearchCV # 9.하이퍼파라미터 튜닝
```
---
### Jupyter Notebook 기본 단축키
* Shift + ENter 키를 누르면 셀이 실행되고 커서가 다음셀로 이동한다.
* Enter 키를 누르면 다시 편집상태로 돌아온다.
* ESC 키를 누르고
    * a키를 누르면 위에 셀 추가
    * b키를 누르면 아래 셀 추가
    * dd키를 누르면 셀 삭제
    * m키를 누르면 마크다운
    * y키를 누르면 코드
    
---
### Markdown 양식
* 챕터 제목은 #
* 부챕터 제목은 ##
* 소챕터 제목은 ###

# Ch02. **통계학습** 
## 2.2 모델의 정확도 평가
## 학습목표
*  fitting, bias-var trade-off)
*  Bayes Classifier
*  KNN
___
### 2.2.1 적합 품질 측정

모델의 정확도를 평가하기 위해 회귀 설정에서 일반적으로 사용되는 평균제곱오차(MSE)

(1) 훈련 MSE
$$MSE=\frac{1}{n}\sum_{i=1}^{n} (y_{i}-\hat{f}(x_{i}))^{2}$$

일반적으로 기계학습을 할때, 훈련(train)데이터와 검정(test)데이터를 나눈다. 이때 (1)의 MSE는 훈련 데이터를 사용하여 계산되기 때문에 정확히는 train MSE 라고 한다.

___
### 2.2.2 편향-분산 trade off
편향과 분산은 절충됨.

**under fitting** : 모델이 너무 단순, **고편향**됨. 모델이 모든 특징을 적절히 설명할 수 있을 만큼 유연성이 충분하지 않다.


**over fitting** : 모델이복잡, **과적합**됨. 모델이 모든 특징을 세밀하게 설명할 수 있을 만큼 유연하지만, 훈련 데이터의 잡음(noise)까지 반영함. 따라서 새로운 데이터가 들어왔을 때 예측오차가 커지는 문제가 발생.

---
### 2.2.3 Model(Bayes Classifier, KNN)

**Bayes Classifier(베이즈 분류기)**

모수적추정 : 주어진 데이터 X에 대한 Y의 조건부분포에 대한 '사전 정보'가 필요

ex) 정규 분포, 포아송 분포, 카이제곱 분포 등등

데이터가 많을 수록 옳은 결정을 할 확률이 올라감. 사전 확률을 지속적으로 업데이트하는 방법.

* Navie Bayes(나이브 베이즈 분류기법) : 추가되는 사후확률을 이어서 곱하여 정보를 추가하는 단순한 기법

**KNN(k-최인접이웃)**
비모수적추정 : 주어진 데이터 X에 대한 Y의 조건부분포에 대한 '사전 정보'가 필요하지 않음.

KNN 분류기는 X에 대한 Y의 조건부분포를 추정하여 가장 높은 추정확률을 가지는 클래스로 관측치를 분류한다.

## **실습**
---
**bias-var trade off** 

overfitting & underfitting
```python
    #package load
from sklearn.preprocessing import PolynomialFeatures # 차원 조정
from sklearn.linear_model import LinearRegression    # 선형 모형
from sklearn.pipeline import make_pipeline           # 파이프라인 패키지
import numpy as np                                   # 기본 패키지

    #random date generate
np.random.seed(1)                                    # 시드 고정
X = np.random.rand(40, 1) ** 2                       # 40행, 1열
y = (10 - 1. / (X.ravel() + 0.1)) + np.random.randn(40) # np.random.randn 을 사용하여 무작위 40개 값 생성

    # plot style package
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

    # 그림
X_test = np.linspace(-0.1, 1.1, 500).reshape(-1,1)   # -0.1에서 1.1 사이에 500개 균등하게 구간 나누고, -1,1의 2차원 reshpae

fig = plt.figure(figsize=(12, 10))
for i, degree in enumerate([1, 3, 5, 10], start=1):  # 1, 3, 5, 10 의 차원(총4개), 1부터 시작하는 인덱스
    ax = fig.add_subplot(2, 2, i)                    # 2,2,1/2,2,2/2,2,3/2,2,4
    ax.scatter(X.ravel(), y, s=15)                   # 산점도 그리기
    y_test = make_pipeline(PolynomialFeatures(degree), LinearRegression())\
             .fit(X, y).predict(X_test)              # 파이프라인 만들기
    ax.plot(X_test.ravel(), y_test, label='degree{0}'.format(degree))
    ax.set_xlim(-0.1, 1.0)
    ax.set_ylim(-2, 12)
    ax.legend(loc='best');
    
    # 검증 곡선
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(1)
X = np.random.rand(40, 1) ** 2
y = (10 - 1. / (X.ravel() + 0.1)) + np.random.randn(40)

from sklearn.model_selection import validation_curve
degree = np.arange(0, 21)

train_score, val_score = (make_pipeline(PolynomialFeatures(degree=2),\
                                        LinearRegression()),\
                          X, y, "polynomialfeatures__degree", degree, cv=7)

plt.figure(figsize=(8, 5))
plt.plot(degree, np.median(train_score, 1), "b-", label="training score")
plt.plot(degree, np.median(val_score, 1), "r-", label="validation score")
plt.ylim(0, 1)
plt.xlabel("degree")
plt.ylabel("score")
plt.legend(loc="best");

    # 검증 곡선에서 degree(복잡도)= 3 
X_test = np.linspace(-0.1, 1.1, 500).reshape(-1, 1)

plt.figure(figsize=(8, 7))
plt.scatter(X.ravel(), y)
lim = plt.axis()
y_pred = make_pipeline(PolynomialFeatures(degree=3), LinearRegression()).fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_pred)
plt.axis(lim);
```


```python
#실습 : overfitting & underfitting
from sklearn.preprocessing import PolynomialFeatures # 차원 조정
from sklearn.linear_model import LinearRegression    # 선형 모형
from sklearn.pipeline import make_pipeline           # 파이프라인 패키지
import numpy as np                                   # 기본 패키지

#random date generate
np.random.seed(1)                                    # 시드 고정
X = np.random.rand(40, 1) ** 2                       # 40행, 1열
y = (10 - 1. / (X.ravel() + 0.1)) + np.random.randn(40) # np.random.randn 을 사용하여 무작위 40개 값 생성

# plot style package
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

# 그림
X_test = np.linspace(-0.1, 1.1, 500).reshape(-1,1)   # -0.1에서 1.1 사이에 500개 균등하게 구간 나누고, -1,1의 2차원 reshpae

fig = plt.figure(figsize=(12, 10))
for i, degree in enumerate([1, 3, 5, 10], start=1):  # 1, 3, 5, 10 의 차원(총4개), 1부터 시작하는 인덱스
    ax = fig.add_subplot(2, 2, i)                    # 2,2,1/2,2,2/2,2,3/2,2,4
    ax.scatter(X.ravel(), y, s=15)                   # 산점도 그리기
    y_test = make_pipeline(PolynomialFeatures(degree), LinearRegression())\
             .fit(X, y).predict(X_test)              # 파이프라인 만들기
    ax.plot(X_test.ravel(), y_test, label='degree{0}'.format(degree))
    ax.set_xlim(-0.1, 1.0)
    ax.set_ylim(-2, 12)
    ax.legend(loc='best');
```


    

    


![python image2](https://github.com/minkim1423/minkim1423.github.io/blob/main/output_3_0.png?raw=true)


```python
# 검증 곡선
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(1)
X = np.random.rand(40, 1) ** 2
y = (10 - 1. / (X.ravel() + 0.1)) + np.random.randn(40)

from sklearn.model_selection import validation_curve
degree = np.arange(0, 21)

train_score, val_score = validation_curve\
(make_pipeline(PolynomialFeatures(degree=2),\
                               LinearRegression()),\
                 X, y, "polynomialfeatures__degree", degree, cv=7)

plt.figure(figsize=(8, 5))
plt.plot(degree, np.median(train_score, 1), "b-", label="training score")
plt.plot(degree, np.median(val_score, 1), "r-", label="validation score")
plt.ylim(0, 1)
plt.xlabel("degree")
plt.ylabel("score")
plt.legend(loc="best");
```

    C:\Users\ahdal\anaconda3\lib\site-packages\sklearn\utils\validation.py:67: FutureWarning: Pass param_name=polynomialfeatures__degree, param_range=[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20] as keyword args. From version 0.25 passing these as positional arguments will result in an error
      warnings.warn("Pass {} as keyword args. From version 0.25 "
    


    
![png](output_5_1.png)
    


![python image2](https://github.com/minkim1423/minkim1423.github.io/blob/main/output_4_1.png?raw=true)


```python
    # 검증 곡선에서 degree(복잡도)= 3 
X_test = np.linspace(-0.1, 1.1, 500).reshape(-1, 1)
plt.figure(figsize=(8, 7))
plt.scatter(X.ravel(), y)
lim = plt.axis()
y_pred = make_pipeline(PolynomialFeatures(degree=3), LinearRegression()).fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_pred)
plt.axis(lim);
```


    
![png](output_7_0.png)
    


![python image2](https://github.com/minkim1423/minkim1423.github.io/blob/main/output_5_0.png?raw=true)

# 예제 : KNN classifier with iris data 

## **예제코드**
---
### 전처리
#### package load
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
```
#### data load and summarize
```python
iris = pd.read_csv('iris.csv')
iris.head()
iris.tail()
iris.shape
iris['variety'].value_counts()
iris.columns
iris.values
iris.info() # stata의 tabulate
iris.describe() # stata의 summarize, detail
iris.describe(include='all')
```
#### dataframe
```python
X = iris.iloc[:,:4] # all row, 4th까지 column을 X에 넣기
X.head()
y = iris.iloc[:,-1] # 모든 row, 맨끝의 column을 y에 넣기
y.head()
X = preprocessing.StandardScaler().fit_transform(X)
X[0:4]
```

---
### 분석
#### Train, Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # test_size=0.3 검증을 30% 데이터로 훈련을 70% 데이터로
y_test.shape
```
#### Training and Predicting
```python
knnmodel=KNeighborsClassifier(n_neighbors=3)
knnmodel.fit(X_train,y_train)
y_predict1=knnmodel.predict(X_test)
```
---
### 검정
#### Accuracy

```python
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_predict1)
acc
```
#### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test.values, y_predict1)
cm
```

```python
cunfmat = pd.DataFrame(confusion_matrix(y_test, y_predict1),
                        index=['True[setosa]','True[versicolor]','True[virginica]'],
                        columns=['Predict[setosa]', 'Predict[versicolor]', 'Predict[virginica]']) # 순서가 중요
cunfmat
```

#### Output Visualization
```python
prediction_output=pd.DataFrame(data=[y_test.values,y_predict1],index=['y_test','y_predict1'])
prediction_output.transpose()
```

```python
prediction_output.iloc[0,:].value_counts()
```

```python
print('잘못 분류된 샘플 개수: %d' % (y_test != y_predict1).sum()) # 뒤에 있는%를 앞에 있는 %에 넣는다는 의미
print('정확도: %.3f' % accuracy_score(y_test, y_predict1)) # 소수점 3째자리 실수로 넣기
```


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
```


```python
iris = pd.read_csv('iris.csv')
iris.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal.length</th>
      <th>sepal.width</th>
      <th>petal.length</th>
      <th>petal.width</th>
      <th>variety</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris.shape
```




    (150, 5)




```python
iris['variety'].value_counts()
```




    Virginica     50
    Setosa        50
    Versicolor    50
    Name: variety, dtype: int64




```python
iris.columns
```




    Index(['sepal.length', 'sepal.width', 'petal.length', 'petal.width',
           'variety'],
          dtype='object')




```python
iris.values
```




    array([[5.1, 3.5, 1.4, 0.2, 'Setosa'],
           [4.9, 3.0, 1.4, 0.2, 'Setosa'],
           [4.7, 3.2, 1.3, 0.2, 'Setosa'],
           [4.6, 3.1, 1.5, 0.2, 'Setosa'],
           [5.0, 3.6, 1.4, 0.2, 'Setosa'],
           [5.4, 3.9, 1.7, 0.4, 'Setosa'],
           [4.6, 3.4, 1.4, 0.3, 'Setosa'],
           [5.0, 3.4, 1.5, 0.2, 'Setosa'],
           [4.4, 2.9, 1.4, 0.2, 'Setosa'],
           [4.9, 3.1, 1.5, 0.1, 'Setosa'],
           [5.4, 3.7, 1.5, 0.2, 'Setosa'],
           [4.8, 3.4, 1.6, 0.2, 'Setosa'],
           [4.8, 3.0, 1.4, 0.1, 'Setosa'],
           [4.3, 3.0, 1.1, 0.1, 'Setosa'],
           [5.8, 4.0, 1.2, 0.2, 'Setosa'],
           [5.7, 4.4, 1.5, 0.4, 'Setosa'],
           [5.4, 3.9, 1.3, 0.4, 'Setosa'],
           [5.1, 3.5, 1.4, 0.3, 'Setosa'],
           [5.7, 3.8, 1.7, 0.3, 'Setosa'],
           [5.1, 3.8, 1.5, 0.3, 'Setosa'],
           [5.4, 3.4, 1.7, 0.2, 'Setosa'],
           [5.1, 3.7, 1.5, 0.4, 'Setosa'],
           [4.6, 3.6, 1.0, 0.2, 'Setosa'],
           [5.1, 3.3, 1.7, 0.5, 'Setosa'],
           [4.8, 3.4, 1.9, 0.2, 'Setosa'],
           [5.0, 3.0, 1.6, 0.2, 'Setosa'],
           [5.0, 3.4, 1.6, 0.4, 'Setosa'],
           [5.2, 3.5, 1.5, 0.2, 'Setosa'],
           [5.2, 3.4, 1.4, 0.2, 'Setosa'],
           [4.7, 3.2, 1.6, 0.2, 'Setosa'],
           [4.8, 3.1, 1.6, 0.2, 'Setosa'],
           [5.4, 3.4, 1.5, 0.4, 'Setosa'],
           [5.2, 4.1, 1.5, 0.1, 'Setosa'],
           [5.5, 4.2, 1.4, 0.2, 'Setosa'],
           [4.9, 3.1, 1.5, 0.2, 'Setosa'],
           [5.0, 3.2, 1.2, 0.2, 'Setosa'],
           [5.5, 3.5, 1.3, 0.2, 'Setosa'],
           [4.9, 3.6, 1.4, 0.1, 'Setosa'],
           [4.4, 3.0, 1.3, 0.2, 'Setosa'],
           [5.1, 3.4, 1.5, 0.2, 'Setosa'],
           [5.0, 3.5, 1.3, 0.3, 'Setosa'],
           [4.5, 2.3, 1.3, 0.3, 'Setosa'],
           [4.4, 3.2, 1.3, 0.2, 'Setosa'],
           [5.0, 3.5, 1.6, 0.6, 'Setosa'],
           [5.1, 3.8, 1.9, 0.4, 'Setosa'],
           [4.8, 3.0, 1.4, 0.3, 'Setosa'],
           [5.1, 3.8, 1.6, 0.2, 'Setosa'],
           [4.6, 3.2, 1.4, 0.2, 'Setosa'],
           [5.3, 3.7, 1.5, 0.2, 'Setosa'],
           [5.0, 3.3, 1.4, 0.2, 'Setosa'],
           [7.0, 3.2, 4.7, 1.4, 'Versicolor'],
           [6.4, 3.2, 4.5, 1.5, 'Versicolor'],
           [6.9, 3.1, 4.9, 1.5, 'Versicolor'],
           [5.5, 2.3, 4.0, 1.3, 'Versicolor'],
           [6.5, 2.8, 4.6, 1.5, 'Versicolor'],
           [5.7, 2.8, 4.5, 1.3, 'Versicolor'],
           [6.3, 3.3, 4.7, 1.6, 'Versicolor'],
           [4.9, 2.4, 3.3, 1.0, 'Versicolor'],
           [6.6, 2.9, 4.6, 1.3, 'Versicolor'],
           [5.2, 2.7, 3.9, 1.4, 'Versicolor'],
           [5.0, 2.0, 3.5, 1.0, 'Versicolor'],
           [5.9, 3.0, 4.2, 1.5, 'Versicolor'],
           [6.0, 2.2, 4.0, 1.0, 'Versicolor'],
           [6.1, 2.9, 4.7, 1.4, 'Versicolor'],
           [5.6, 2.9, 3.6, 1.3, 'Versicolor'],
           [6.7, 3.1, 4.4, 1.4, 'Versicolor'],
           [5.6, 3.0, 4.5, 1.5, 'Versicolor'],
           [5.8, 2.7, 4.1, 1.0, 'Versicolor'],
           [6.2, 2.2, 4.5, 1.5, 'Versicolor'],
           [5.6, 2.5, 3.9, 1.1, 'Versicolor'],
           [5.9, 3.2, 4.8, 1.8, 'Versicolor'],
           [6.1, 2.8, 4.0, 1.3, 'Versicolor'],
           [6.3, 2.5, 4.9, 1.5, 'Versicolor'],
           [6.1, 2.8, 4.7, 1.2, 'Versicolor'],
           [6.4, 2.9, 4.3, 1.3, 'Versicolor'],
           [6.6, 3.0, 4.4, 1.4, 'Versicolor'],
           [6.8, 2.8, 4.8, 1.4, 'Versicolor'],
           [6.7, 3.0, 5.0, 1.7, 'Versicolor'],
           [6.0, 2.9, 4.5, 1.5, 'Versicolor'],
           [5.7, 2.6, 3.5, 1.0, 'Versicolor'],
           [5.5, 2.4, 3.8, 1.1, 'Versicolor'],
           [5.5, 2.4, 3.7, 1.0, 'Versicolor'],
           [5.8, 2.7, 3.9, 1.2, 'Versicolor'],
           [6.0, 2.7, 5.1, 1.6, 'Versicolor'],
           [5.4, 3.0, 4.5, 1.5, 'Versicolor'],
           [6.0, 3.4, 4.5, 1.6, 'Versicolor'],
           [6.7, 3.1, 4.7, 1.5, 'Versicolor'],
           [6.3, 2.3, 4.4, 1.3, 'Versicolor'],
           [5.6, 3.0, 4.1, 1.3, 'Versicolor'],
           [5.5, 2.5, 4.0, 1.3, 'Versicolor'],
           [5.5, 2.6, 4.4, 1.2, 'Versicolor'],
           [6.1, 3.0, 4.6, 1.4, 'Versicolor'],
           [5.8, 2.6, 4.0, 1.2, 'Versicolor'],
           [5.0, 2.3, 3.3, 1.0, 'Versicolor'],
           [5.6, 2.7, 4.2, 1.3, 'Versicolor'],
           [5.7, 3.0, 4.2, 1.2, 'Versicolor'],
           [5.7, 2.9, 4.2, 1.3, 'Versicolor'],
           [6.2, 2.9, 4.3, 1.3, 'Versicolor'],
           [5.1, 2.5, 3.0, 1.1, 'Versicolor'],
           [5.7, 2.8, 4.1, 1.3, 'Versicolor'],
           [6.3, 3.3, 6.0, 2.5, 'Virginica'],
           [5.8, 2.7, 5.1, 1.9, 'Virginica'],
           [7.1, 3.0, 5.9, 2.1, 'Virginica'],
           [6.3, 2.9, 5.6, 1.8, 'Virginica'],
           [6.5, 3.0, 5.8, 2.2, 'Virginica'],
           [7.6, 3.0, 6.6, 2.1, 'Virginica'],
           [4.9, 2.5, 4.5, 1.7, 'Virginica'],
           [7.3, 2.9, 6.3, 1.8, 'Virginica'],
           [6.7, 2.5, 5.8, 1.8, 'Virginica'],
           [7.2, 3.6, 6.1, 2.5, 'Virginica'],
           [6.5, 3.2, 5.1, 2.0, 'Virginica'],
           [6.4, 2.7, 5.3, 1.9, 'Virginica'],
           [6.8, 3.0, 5.5, 2.1, 'Virginica'],
           [5.7, 2.5, 5.0, 2.0, 'Virginica'],
           [5.8, 2.8, 5.1, 2.4, 'Virginica'],
           [6.4, 3.2, 5.3, 2.3, 'Virginica'],
           [6.5, 3.0, 5.5, 1.8, 'Virginica'],
           [7.7, 3.8, 6.7, 2.2, 'Virginica'],
           [7.7, 2.6, 6.9, 2.3, 'Virginica'],
           [6.0, 2.2, 5.0, 1.5, 'Virginica'],
           [6.9, 3.2, 5.7, 2.3, 'Virginica'],
           [5.6, 2.8, 4.9, 2.0, 'Virginica'],
           [7.7, 2.8, 6.7, 2.0, 'Virginica'],
           [6.3, 2.7, 4.9, 1.8, 'Virginica'],
           [6.7, 3.3, 5.7, 2.1, 'Virginica'],
           [7.2, 3.2, 6.0, 1.8, 'Virginica'],
           [6.2, 2.8, 4.8, 1.8, 'Virginica'],
           [6.1, 3.0, 4.9, 1.8, 'Virginica'],
           [6.4, 2.8, 5.6, 2.1, 'Virginica'],
           [7.2, 3.0, 5.8, 1.6, 'Virginica'],
           [7.4, 2.8, 6.1, 1.9, 'Virginica'],
           [7.9, 3.8, 6.4, 2.0, 'Virginica'],
           [6.4, 2.8, 5.6, 2.2, 'Virginica'],
           [6.3, 2.8, 5.1, 1.5, 'Virginica'],
           [6.1, 2.6, 5.6, 1.4, 'Virginica'],
           [7.7, 3.0, 6.1, 2.3, 'Virginica'],
           [6.3, 3.4, 5.6, 2.4, 'Virginica'],
           [6.4, 3.1, 5.5, 1.8, 'Virginica'],
           [6.0, 3.0, 4.8, 1.8, 'Virginica'],
           [6.9, 3.1, 5.4, 2.1, 'Virginica'],
           [6.7, 3.1, 5.6, 2.4, 'Virginica'],
           [6.9, 3.1, 5.1, 2.3, 'Virginica'],
           [5.8, 2.7, 5.1, 1.9, 'Virginica'],
           [6.8, 3.2, 5.9, 2.3, 'Virginica'],
           [6.7, 3.3, 5.7, 2.5, 'Virginica'],
           [6.7, 3.0, 5.2, 2.3, 'Virginica'],
           [6.3, 2.5, 5.0, 1.9, 'Virginica'],
           [6.5, 3.0, 5.2, 2.0, 'Virginica'],
           [6.2, 3.4, 5.4, 2.3, 'Virginica'],
           [5.9, 3.0, 5.1, 1.8, 'Virginica']], dtype=object)




```python
iris.info() 
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   sepal.length  150 non-null    float64
     1   sepal.width   150 non-null    float64
     2   petal.length  150 non-null    float64
     3   petal.width   150 non-null    float64
     4   variety       150 non-null    object 
    dtypes: float64(4), object(1)
    memory usage: 6.0+ KB
    


```python
iris.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal.length</th>
      <th>sepal.width</th>
      <th>petal.length</th>
      <th>petal.width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.843333</td>
      <td>3.057333</td>
      <td>3.758000</td>
      <td>1.199333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828066</td>
      <td>0.435866</td>
      <td>1.765298</td>
      <td>0.762238</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.350000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal.length</th>
      <th>sepal.width</th>
      <th>petal.length</th>
      <th>petal.width</th>
      <th>variety</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>50</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.843333</td>
      <td>3.057333</td>
      <td>3.758000</td>
      <td>1.199333</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828066</td>
      <td>0.435866</td>
      <td>1.765298</td>
      <td>0.762238</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.350000</td>
      <td>1.300000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = iris.iloc[:,:4] 
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal.length</th>
      <th>sepal.width</th>
      <th>petal.length</th>
      <th>petal.width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = iris.iloc[:,-1]
y.head()
```




    0    Setosa
    1    Setosa
    2    Setosa
    3    Setosa
    4    Setosa
    Name: variety, dtype: object




```python
X = preprocessing.StandardScaler().fit_transform(X)
X[0:4]
```




    array([[-0.90068117,  1.01900435, -1.34022653, -1.3154443 ],
           [-1.14301691, -0.13197948, -1.34022653, -1.3154443 ],
           [-1.38535265,  0.32841405, -1.39706395, -1.3154443 ],
           [-1.50652052,  0.09821729, -1.2833891 , -1.3154443 ]])



# Train, Test Split


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
y_test.shape
```




    (45,)



# Training and Predicting


```python
knnmodel=KNeighborsClassifier(n_neighbors=3)
knnmodel.fit(X_train,y_train)
y_predict1=knnmodel.predict(X_test)
```

# Accuracy


```python
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_predict1)
acc
```




    0.9777777777777777



# Confusion Matrix


```python
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test.values, y_predict1)
cm
```




    array([[14,  0,  0],
           [ 0, 18,  0],
           [ 0,  1, 12]], dtype=int64)




```python
cunfmat = pd.DataFrame(confusion_matrix(y_test, y_predict1),
                        index=['True[setosa]','True[versicolor]','True[virginica]'],
                        columns=['Predict[setosa]', 'Predict[versicolor]', 'Predict[virginica]']) # 순서가 중요
cunfmat
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predict[setosa]</th>
      <th>Predict[versicolor]</th>
      <th>Predict[virginica]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>True[setosa]</th>
      <td>14</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>True[versicolor]</th>
      <td>0</td>
      <td>18</td>
      <td>0</td>
    </tr>
    <tr>
      <th>True[virginica]</th>
      <td>0</td>
      <td>1</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>



# Output Visualization


```python
prediction_output=pd.DataFrame(data=[y_test.values,y_predict1],index=['y_test','y_predict1'])
prediction_output.transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y_test</th>
      <th>y_predict1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Setosa</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Versicolor</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Versicolor</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Setosa</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Virginica</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Versicolor</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Virginica</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Setosa</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Setosa</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Virginica</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Versicolor</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Setosa</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Virginica</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Versicolor</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Versicolor</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Setosa</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Versicolor</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Versicolor</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Setosa</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Setosa</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Versicolor</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Versicolor</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Versicolor</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Setosa</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Virginica</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Versicolor</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Setosa</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Setosa</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Versicolor</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Virginica</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Versicolor</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Virginica</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Versicolor</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Virginica</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Virginica</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Setosa</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Versicolor</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Setosa</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Versicolor</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Virginica</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Virginica</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Setosa</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Virginica</td>
      <td>Versicolor</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Virginica</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Versicolor</td>
      <td>Versicolor</td>
    </tr>
  </tbody>
</table>
</div>




```python
prediction_output.iloc[0,:].value_counts()
```




    Versicolor    18
    Setosa        14
    Virginica     13
    Name: y_test, dtype: int64




```python
print('잘못 분류된 샘플 개수: %d' % (y_test != y_predict1).sum()) # 뒤에 있는%를 앞에 있는 %에 넣는다는 의미
print('정확도: %.3f' % accuracy_score(y_test, y_predict1)) # 소수점 3째자리 실수로 넣기
```

    잘못 분류된 샘플 개수: 1
    정확도: 0.978
    

    * TPR = True Positive Rate : 민감도(sensitvity) = 재현율(Recall) = 3/4
    실제값 중에 잘 맞춘것 = TP/P
    * FPR = False Positive Rate 
    예측값 중에 잘 맞춘것 = FP/N
    *민감도(=TPR)와 특이도(=1-FPR)

    *정밀도=TP/(TP+FP)=3/4
* True : TN/TP, True Negative, True Positive
* Predict : PN/PP,

* F1 = 2*(재현율*정밀도)/(재현율+정밀도)=2*(0.75 * 0.75)/(0.75+0.75)=0.75
