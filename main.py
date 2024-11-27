import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv("data.csv")
newdf=df.drop(['Unnamed: 32', 'id'],axis = 1)

def diagnosis_value(diagnosis):
    if diagnosis== 'M':
        return 1
    else:
        return 0
newdf['diagnosis'] = newdf['diagnosis'].apply(diagnosis_value)

# sns.lmplot(x = 'radius_mean', y = 'texture_mean', hue = 'diagnosis', data = newdf) 
# sns.lmplot(x ='smoothness_mean', y = 'compactness_mean', data = newdf, hue = 'diagnosis') 

X = np.array(newdf.iloc[:, 1:]) 
y = np.array(newdf['diagnosis']) 

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

knn=KNeighborsClassifier(n_neighbors=12)
knn.fit(x_train,y_train)


score=knn.score(x_test, y_test)

neighbors = [] 
cv_scores = [] 
  

# perform 10 fold cross validation 
for k in range(1, 51, 2): 
    neighbors.append(k) 
    knn = KNeighborsClassifier(n_neighbors = k) 
    scores = cross_val_score( 
        knn, x_train, y_train, cv = 10, scoring = 'accuracy') 
    cv_scores.append(scores.mean()) 


print(cv_scores)

MSE = [1-x for x in cv_scores] 
plt.figure(figsize = (10, 6)) 
plt.plot(neighbors, cv_scores) 
plt.xlabel('Number of neighbors') 
plt.ylabel('cv scores') 
plt.show()
  
# determining the best k 
# optimal_k = neighbors[MSE.index(min(MSE))] 
# print('The optimal number of neighbors is % d ' % optimal_k) 
  
# # plot misclassification error versus k 
# plt.figure(figsize = (10, 6)) 
# plt.plot(neighbors, MSE) 
# plt.xlabel('Number of neighbors') 
# plt.ylabel('Misclassification Error') 
# plt.show() 
