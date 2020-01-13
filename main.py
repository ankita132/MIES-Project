import numpy as np
import pandas as pd
import extractor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore", DeprecationWarning)

# data extraction
X_user1 = extractor.features(34,"data/16EC10004/")
X_user1= pd.DataFrame(X_user1)
X_user1['Class']=0
print('\nDataset 1 extracted')

X_user2 = extractor.features(20,"data/16EC10022/")
X_user2= pd.DataFrame(X_user2)
X_user2['Class']=1
print('\nDataset 2 extracted')

X_user3 = extractor.features(26,"data/16EC35020/")
X_user3 = pd.DataFrame(X_user3)
X_user3['Class']=2
print('\nDataset 3 extracted')

X_user4 = extractor.features(26,"data/16EC35024/")
X_user4 = pd.DataFrame(X_user4)
X_user4['Class']=3
print('\nDataset 4 extracted')

X_user5 = extractor.features(28,"data/16EC35026/")
X_user5 = pd.DataFrame(X_user5)
X_user5['Class']= 4
print('\nDataset 5 extracted')

X_user6 = extractor.features(41,"data/18CH10058/")
X_user6 = pd.DataFrame(X_user6)
X_user6['Class']= 5
print('\nDataset 6 extracted')

X_user7 = extractor.features(23,"data/18EC10070/")
X_user7 = pd.DataFrame(X_user7)
X_user7['Class']= 6
print('\nDataset 7 extracted')

X_user8 = extractor.features(13,"data/18EC10074/")
X_user8 = pd.DataFrame(X_user8)
X_user8['Class']= 7
print('\nDataset 8 extracted')

X_user9 = extractor.features(26,"data/18EC30002/")
X_user9 = pd.DataFrame(X_user9)
X_user9['Class']= 8
print('\nDataset 9 extracted')

X_user10 = extractor.features(25,"data/18EC30034/")
X_user10 = pd.DataFrame(X_user10)
X_user10['Class']= 9
print('\nDataset 10 extracted')

print('\nData Extraction Complete.')

#defining training and testing data
X_user1_train = X_user1[:int(X_user1.shape[0]*0.8)]
X_user1_test = X_user1[int(X_user1.shape[0]*0.8):]

X_user2_train = X_user2[:int(X_user2.shape[0]*0.8)]
X_user2_test = X_user2[int(X_user2.shape[0]*0.8):]

X_user3_train = X_user3[:int(X_user3.shape[0]*0.8)]
X_user3_test = X_user3[int(X_user3.shape[0]*0.8):]

X_user4_train = X_user4[:int(X_user4.shape[0]*0.8)]
X_user4_test = X_user4[int(X_user4.shape[0]*0.8):]

X_user5_train = X_user5[:int(X_user5.shape[0]*0.8)]
X_user5_test = X_user5[int(X_user5.shape[0]*0.8):]

X_user6_train = X_user6[:int(X_user6.shape[0]*0.8)]
X_user6_test = X_user6[int(X_user6.shape[0]*0.8):]

X_user7_train = X_user7[:int(X_user7.shape[0]*0.8)]
X_user7_test = X_user7[int(X_user7.shape[0]*0.8):]

X_user8_train = X_user8[:int(X_user8.shape[0]*0.8)]
X_user8_test = X_user8[int(X_user8.shape[0]*0.8):]

X_user9_train = X_user9[:int(X_user9.shape[0]*0.8)]
X_user9_test = X_user9[int(X_user9.shape[0]*0.8):]

X_user10_train = X_user10[:int(X_user10.shape[0]*0.8)]
X_user10_test = X_user10[int(X_user10.shape[0]*0.8):]

# X_train=X_user1_train.append([X_user3_train])
# X_test=X_user1_test.append([X_user3_test])

X_train=X_user1_train.append([X_user2_train,X_user3_train,X_user4_train,X_user5_train,X_user6_train,X_user7_train,X_user8_train,X_user9_train,X_user10_train])
X_test=X_user1_test.append([X_user2_test,X_user3_test,X_user4_test,X_user5_test,X_user6_test,X_user7_test,X_user8_test,X_user9_test,X_user10_test])

y_train=X_train[['Class']]
y_test=X_test[['Class']]
X_data=X_train.append(X_test)
X_data = X_data.reset_index(drop=True)
y_data=y_train.append(y_test)
y_data = y_data.reset_index(drop=True)
X_data.to_csv('file_X.csv') 
y_data.to_csv('file_Y.csv')
print('\nPre-processing Done.')

print('\nCount of different classes in Train set:')
print(X_train['Class'].value_counts())

print('\nCount of different classes in Test set:')
print(X_test['Class'].value_counts())

feats=[c for c in X_train.columns if c!='Class']

# Train classifier
print('\nImplementing Gaussian Naive Bayes Model.')
gnb = GaussianNB()
gnb.fit(
    X_train[feats].values,
    y_train['Class']
)
y_pred = gnb.predict(X_test[feats].values)

print("\nNumber of mislabeled points out of a total {} points : {}, Accuracy: {:05.5f}%"
      .format(
          X_test.shape[0],
          (X_test["Class"] != y_pred).sum(),
          100*(1-(X_test["Class"] != y_pred).sum()/X_test.shape[0])
))

#five fold cross validation
cv = KFold(n_splits=5)
clf = GaussianNB()
X_data=X_data.values
y_data=y_data.values
accuracy=0
count = 1
for traincv, testcv in cv.split(X_data):
        clf.fit(X_data[traincv], y_data[traincv])
        train_predictions = clf.predict(X_data[testcv])
        acc = accuracy_score(y_data[testcv], train_predictions)
        print("{} Fold Cross Validation Accuracy : {}".format(count, acc))
        print("{} Fold Cross Validation F1 Score : {}".format(count, f1_score(y_data[testcv], train_predictions, average="macro")))
        print("{} Fold Cross Validation Recall : {}".format(count, recall_score(y_data[testcv], train_predictions, average="macro")))
        print("{} Fold Cross Validation Precision : {}".format(count, precision_score(y_data[testcv], train_predictions, average="macro")))
        count = count + 1
        accuracy+= acc
       
accuracy = 20*accuracy
print('\n5 Fold Cross Validation Accuracy on Training Set: '+str(accuracy))
