"""
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import roc_auc_score,log_loss
from sklearn.ensemble import RandomForestClassifier


#輸入資料
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

#Logistic Regression

trainLoss=[]
valiLoss=[]
models=[]

for trainIndex,valIndex in kf.split(X_train):
    X_fold_train,X_fold_val = X_train[trainIndex], X_train[valIndex]  #根據折數切出每一份特徵
    y_fold_train,y_fold_val = y_train.iloc[trainIndex], y_train.iloc[valIndex]  #根據折數切出每一份預測結果
    
    model = RandomForestClassifier()
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
print(f"Randomforest Test log_loss: {test_log_loss_score:.4f}")
print(f"Randomforest Test AUC: {test_auc_score:.4f}")
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.ensemble import RandomForestClassifier

# 載入資料
df = pd.read_csv("HW2/ncku-cs-data-mining-homework-2/train.csv")
df = df.iloc[:, 1:]

# 特徵與標籤
X = df.iloc[:, :22]
Y = df["smoking"]

# 分割訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

# 特徵標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 設定模型
model = RandomForestClassifier(
    random_state=4,
    n_estimators=400,
    max_depth=15,
    min_samples_leaf=3,
    min_samples_split=3
    )

# 超參數網格
param_grid = {
    #'n_estimators': [200,250,300,400],
    #'max_depth': [5, 10, None,15],
    #'min_samples_split': [3,4,5,6,7,8,9,10],
    #'min_samples_leaf': [1, 2,3,4]
}

# 使用 StratifiedKFold 保持類別平衡
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 設定 GridSearch
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# 執行 Grid Search
grid_search.fit(X_train, y_train)

# 最佳模型
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best CV AUC:", grid_search.best_score_)

# 測試集表現
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
test_log_loss_score = log_loss(y_test, y_pred_proba)
test_auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"RandomForest Test log_loss: {test_log_loss_score:.4f}")
print(f"RandomForest Test AUC: {test_auc_score:.4f}")

# 載入測試資料，進行預測
testcsv = pd.read_csv("HW2/ncku-cs-data-mining-homework-2/test.csv")
test_feature = testcsv.iloc[:, 1:]
print("測試特徵的大小:", test_feature.shape)

# 預測
answer = best_model.predict(test_feature)

# 產生提交檔案
submission = pd.DataFrame({
    "id": testcsv.iloc[:, 0],
    "smoking": answer
})
submission.to_csv("HW2/answer_random_forest.csv", index=False)
print("成功生成 answer_randomforest.csv！")

