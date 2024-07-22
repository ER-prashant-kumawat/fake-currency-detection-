# fake-currency-detection-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('banknotes.csv')
df.head()
df.describe()
df.info()
sns.heatmap(df.isnull())
plt.title("Missing values?",fontsize = 18)
plt.show()
sns.pairplot(df, hue = "conterfeit")
plt.show()
sns.countplot(x='conterfeit',data = df)
sns.distplot(df.conterfeit)
sns.barplot(x = 'Right',y = 'Left',data = df,hue= 'conterfeit')
sns.boxplot(x = 'Right',y = 'Left',data=df,hue ='conterfeit')
sns.boxplot(x ='Top',y ='Bottom',data = df,hue ='conterfeit')
sns.barplot(x = 'Top',y = 'Bottom',data = df,hue = 'conterfeit')
sns.heatmap(df.corr(),annot = True,cmap="RdBu")
plt.title("Pairwise correlation of the columns",fontsize = 18)
plt.show()
df = df.reindex(np.random.permutation(df.index))

x = df.drop(columns = "conterfeit")
y = df["conterfeit"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.preprocessing import StandardScaler
st = StandardScaler()
x_train = st.fit_transform(x_train)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)

pred = model.predict(st.transform(x_test))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class_report = classification_report(y_test,pred)
conf_matrix = confusion_matrix(y_test,pred)
acc = accuracy_score(y_test,pred)

print("Classification report:\n\n", class_report)
print("confusion Matrix\n",conf_matrix)
print("\nAccuracy\n", acc)

results = []
results.append(("LogisticRegression",class_report,conf_matrix, acc))
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)

pred = svc.predict(st.transform(x_test))

class_report = classification_report(y_test,pred)
conf_matrix = confusion_matrix(y_test,pred)
acc = accuracy_score(y_test,pred)

print("Classification report:\n\n",class_report)
print("Confusoon Matrix\n",conf_matrix)
print("\nAccuracy\n", acc)
results.append(("SVC", class_report,conf_matrix,acc))

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components = 2,random_state = 0)

transf = svd.fit_transform(x)

plt.scatter(x = transf[:,0],y = transf[:,1])
plt.title("dataset after transformation with SVD", fontsize = 18)
plt.show()

from sklearn.cluster import KMeans

km = KMeans(n_clusters = 2)
c = km.fit_predict(transf)

plt.scatter(x = transf[:,0],y = transf[:,1],c = c)
plt.title("Clustering with Kmeans sfter SVD", fontsize = 18)
plt.show()

plt.scatter(x = transf[:,0],y = transf[:,1],c = y)
plt.title("Original labels after SVD", fontsize = 18)
plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components = 2,random_state = 0)

transf = pca.fit_transform(x)

plt.scatter(x = transf[:,0],y = transf[:,1])
plt.title("Dataset after transformation with PCA", fontsize = 18)
plt.show()

km = KMeans(n_clusters = 2)
c = km.fit_predict(transf)

plt.scatter(x = transf[:,0],y = transf[:,1],c = c)
plt.title("Clustering with Kmeabs after PCA", fontsize = 18)
plt.show()

plt.scatter(x = transf[:,0], y = transf[:,1], c = y)
plt.title("Original labels after PCA", fontsize = 18)
plt.show()

labels  = []
height = []
for i in range(len(results)):
    labels.append(results[i][0])
    height.append(results[i][-1])
    
plt.figure(figsize = (12,6))    
ax = sns.barplot(labels,height)
ax.set_xticklabels(labels, fontsize = 18, rotation = 90)
plt.title("Comparison of the models", fontsize = 18)
plt.ylabel("Prediction accuracy")
plt.show()
