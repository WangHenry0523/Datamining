import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import roc_auc_score,log_loss

df=pd.read_csv("HW2/ncku-cs-data-mining-homework-2/train.csv")
df=df.iloc[:,1:]
#print(df)

#設定特徵與預測結果
X=df.iloc[:,:22]
Y=df["smoking"]
print("特徵 X 的 shape:", X.shape)
print("標籤 Y 的 shape:", Y.shape)


#切分資料集
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True)

#特徵標準化
scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)  #訓練集標準化
X_test= scaler.transform(X_test)  #測試集標準化

#設定交叉驗證折數
kf = KFold(n_splits=5,shuffle=True,random_state=42)

#Decision tree

trainLoss=[]
valiLoss=[]
models=[]

for trainIndex,valIndex in kf.split(X_train):
    X_fold_train,X_fold_val = X_train[trainIndex], X_train[valIndex]  #根據折數切出每一份特徵
    y_fold_train,y_fold_val = y_train.iloc[trainIndex], y_train.iloc[valIndex]  #根據折數切出每一份預測結果
    
    model=DecisionTreeClassifier(random_state=42) #設定模型
    model.fit(X_fold_train,y_fold_train)    #訓練模型
    
    # 計算loss
    train_loss=log_loss(y_fold_train, model.predict_proba(X_fold_train)[:, 1])
    val_loss= log_loss(y_fold_val, model.predict_proba(X_fold_val)[:, 1])
    # 填入loss串列
    trainLoss.append(train_loss)
    valiLoss.append(val_loss)
    #填入models串列
    models.append(model)


# 找出哪個 index 的 loss 最接近中位數
middle_index = np.argmin([abs(loss - (np.median(valiLoss))) for loss in valiLoss])

best_model = models[middle_index]

# 在測試集上計算 log_loss
test_log_loss_score = log_loss(y_test, best_model.predict_proba(X_test)[:, 1])
test_auc_score=roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

# 顯示結果
#print(f"Train log_loss (每折): {trainLoss}")
#print(f"Validation log_loss (每折): {valiLoss}")
print(f"DecisionTree模型 Test log_loss: {test_log_loss_score:.4f}")
print(f"DecisionTree模型 Test AUC: {test_auc_score:.4f}")
