import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import roc_auc_score,log_loss

#輸入資料
df=pd.read_csv("HW2/ncku-cs-data-mining-homework-2/train_with_features.csv")
df=df.iloc[:,1:]

#處理離群值
median=df.median()
for column in df.columns:
    mean=df[column].mean() #計算該column之平均數
    Std=df[column].std()   #計算該column之標準差
    outliers=(df[column]-mean).abs()>3*Std #標記該column中之離群值(減去平均數後取絕對值大於三個標準差者)
    #df[column][outliers]=np.nan
    df.loc[outliers, column] = np.nan  #將標記值替換為NaN
nanCounts = df.isna().sum()  #計算NaN之數量
#print("\nNumber of NaN values for each feature after replacing outliers:\n", nanCounts)
df=df.fillna(median) #將NaN值替換為中位數

#設定特徵與預測結果
X=df.iloc[:,:-1]
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

#Logistic Regression

trainLoss=[]
valiLoss=[]
models=[]

for trainIndex,valIndex in kf.split(X_train):
    X_fold_train,X_fold_val = X_train[trainIndex], X_train[valIndex]  #根據折數切出每一份特徵
    y_fold_train,y_fold_val = y_train.iloc[trainIndex], y_train.iloc[valIndex]  #根據折數切出每一份預測結果
    
    model=linear_model.LogisticRegression(penalty="l2",C=0.1,solver="liblinear",max_iter=1000,class_weight='balanced') #設定模型
    model.fit(X_fold_train,y_fold_train)    #訓練模型
    
    # 計算loss
    train_loss=log_loss(y_fold_train, model.predict_proba(X_fold_train)[:, 1])
    val_loss= log_loss(y_fold_val, model.predict_proba(X_fold_val)[:, 1])
    # 填入loss串列
    trainLoss.append(train_loss)
    valiLoss.append(val_loss)
    #填入models串列
    models.append(model)

# 找到最中間的那一折
middle_index = np.argmin([abs(loss - (np.median(valiLoss))) for loss in valiLoss])

best_model = models[middle_index]

# 在測試集上計算 log_loss
test_log_loss_score = log_loss(y_test, best_model.predict_proba(X_test)[:, 1])
test_auc_score=roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

# 顯示結果
#print(f"Train log_loss (每折): {trainLoss}")
#print(f"Validation log_loss (每折): {valiLoss}")
print(f"Logistic模型 Test log_loss: {test_log_loss_score:.4f}")
print(f"Logistic模型 Test AUC: {test_auc_score:.4f}")

print("==================================================================")
#寫入預測結果
testcsv=pd.read_csv("HW2/ncku-cs-data-mining-homework-2/test_with_features.csv")
test_feature=testcsv.iloc[:,1:]
print("測試特徵的大小",test_feature.shape)
test_feature = scaler.transform(test_feature)
answer=best_model.predict(test_feature)

# 生成提交格式（假設 ID 是 test.csv 裡第一欄）
submission = pd.DataFrame({
    "id": testcsv.iloc[:,0], # 假設第一欄是 id
    "smoking": answer        # 預測的機率
})

# 寫入 CSV（不寫 index）
submission.to_csv("HW2/answer_logistic.csv", index=False)

print("成功生成 answer.csv！")