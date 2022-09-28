import numpy as np
from math import ceil
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

F,N = list(map(int,input().split()))
features = []
target = []
for n in range(N):
    value = list(map(float,input().split()))
    target.append(value[-1])
    feature = []
    for f in range(F):
        feature.append(value[f])
    features.append(feature)

polynomial = PolynomialFeatures(degree=F, include_bias=False)
features_polynomial = polynomial.fit_transform(features)
    
target = np.array(target)
    
regression = LinearRegression()
model = regression.fit(features_polynomial,target)

T = int(input())
for _ in range(T):
    new_value = list(map(float,input().split()))
    new_value = polynomial.fit_transform([new_value])
    number = model.predict(new_value)[0]
    print( ceil(number * 100) / 100.0 )
