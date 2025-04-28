import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import roc_auc_score,log_loss
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#輸入資料
df=pd.read_csv("HW2/ncku-cs-data-mining-homework-2/train.csv")
df=df.iloc[:,1:]
#print(df)

#設定特徵與預測結果
X=df.iloc[:,:-1]
Y=df["smoking"]

print("特徵 X 的 shape:", X.shape)
print("標籤 Y 的 shape:", Y.shape)

#切分資料集
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=32,shuffle=True)

#特徵標準化
#scaler = StandardScaler()
#X_train= scaler.fit_transform(X_train)  #訓練集標準化
#X_test= scaler.transform(X_test)  #測試集標準化

#設定交叉驗證折數
kf = KFold(n_splits=5,shuffle=True,random_state=32)

trainAUC=[]
valiAUC=[]
models=[]

# 建立模型（先不要給參數）
xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False)

# 設定要搜尋的參數組合
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [3, 4],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'reg_alpha': [0.0, 0.1],
    'reg_lambda': [1.0, 5.0]
}

# 建立 GridSearch 物件（用 AUC 當作評分標準）
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    verbose=1,
    n_jobs=-1  # 使用全部 CPU 加速
)

# 執行搜尋
grid_search.fit(X_train, y_train)

# 印出最佳參數與最佳分數
print("最佳參數：", grid_search.best_params_)
print("最佳訓練 AUC：", grid_search.best_score_)

# 最佳模型
best_model = grid_search.best_estimator_

# 在測試集上計算 log_loss & AUC
test_log_loss_score = log_loss(y_test, best_model.predict_proba(X_test)[:, 1])
test_auc_score = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

print(f"XG_Boost Test log_loss: {test_log_loss_score:.4f}")
print(f"XG_Boost Test AUC: {test_auc_score:.4f}")

testcsv = pd.read_csv("HW2/ncku-cs-data-mining-homework-2/test.csv")
test_feature = testcsv.iloc[:, 1:]
answer = best_model.predict(test_feature)

submission = pd.DataFrame({
    "id": testcsv.iloc[:, 0],
    "smoking": answer
})

submission.to_csv("HW2/answer_xgboostv2.csv", index=False)
print("成功生成 answer.csv！")
