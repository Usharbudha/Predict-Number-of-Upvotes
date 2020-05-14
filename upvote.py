import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

train.drop(['Username'],inplace=True,axis=1)
test.drop(['Username'],inplace=True,axis=1)

plt.figure(figsize=(6,4))
cmap=train.corr()
sns.heatmap(cmap,annot=True)

from sklearn.preprocessing import LabelEncoder,StandardScaler,PolynomialFeatures
le=LabelEncoder()
sc=StandardScaler()

train['Tag']=le.fit_transform(train['Tag'])
test['Tag']=le.fit_transform(test['Tag'])

from sklearn.model_selection import train_test_split
from sklearn import linear_model

feature_names=[x for x in train.columns if x not in ['Upvotes']]
y=train['Upvotes']

x_train,x_test,y_train,y_test=train_test_split(train[feature_names],y,test_size=0.2,random_state=250)

x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

poly_reg=PolynomialFeatures(degree=2,interaction_only=False,include_bias=True)
X_poly=poly_reg.fit_transform(x_train)
poly_reg.fit(x_train,y_train)
lin_reg=linear_model.LassoLars(alpha=0.015,max_iter=4000)
lin_reg.fit(X_poly,y_train)

from sklearn.metrics import r2_score
pred=poly_reg.fit_transform(x_test)
y_pred=lin_reg.predict(pred)
print(r2_score(y_test,y_pred))

test=sc.fit_transform(test)
pred_test=poly_reg.fit_transform(test)
pred_imp=lin_reg.predict(pred_test)
pred_imp=abs(pred_imp)

submission=pd.DataFrame({'ID':test['ID'],'Upvotes':pred_imp})
submission.to_csv('SUBMISSION.csv',index=False)
























