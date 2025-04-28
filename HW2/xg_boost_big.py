import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import roc_auc_score,log_loss
from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt

#輸入資料
df=pd.read_csv("HW2/ncku-cs-data-mining-homework-2/train_dataset.csv")
df=df.iloc[:,1:]
#print(df)

#設定特徵與預測結果
X=df.iloc[:,:-1]
Y=df["smoking"]

print("特徵 X 的 shape:", X.shape)
print("標籤 Y 的 shape:", Y.shape)

#切分資料集
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=32,shuffle=True)

#設定交叉驗證折數
kf = KFold(n_splits=5,shuffle=True,random_state=32)

trainAUC=[]
valiAUC=[]
models=[]

for trainIndex,valIndex in kf.split(X_train):
    X_fold_train,X_fold_val = X_train.iloc[trainIndex], X_train.iloc[valIndex]  #根據折數切出每一份特徵
    y_fold_train,y_fold_val = y_train.iloc[trainIndex], y_train.iloc[valIndex]  #根據折數切出每一份預測結果
    
    model = XGBClassifier(
    n_estimators=200,       # 樹的數量
    max_depth=4,            # 每棵樹的最大深度（避免 overfitting）
    learning_rate=0.1,     # 學習率
    subsample=0.8,          # 每棵樹訓練時隨機抽樣比例
    colsample_bytree=0.8,   # 每棵樹隨機選特徵的比例
    reg_alpha=0.1,          # L1 正則化
    reg_lambda=1.0,         # L2 正則化
    eval_metric='logloss'   # 避免 warning
    )
    model.fit(X_fold_train,y_fold_train)    #訓練模型
    
    # 計算AUC
    train_auc=roc_auc_score(y_fold_train, model.predict_proba(X_fold_train)[:, 1])
    val_auc= roc_auc_score(y_fold_val, model.predict_proba(X_fold_val)[:, 1])
    # 填入AUC串列
    trainAUC.append(train_auc)
    valiAUC.append(val_auc)
    #填入models串列
    models.append(model)

# 找到最中間的那一折
middle_index = np.argmax(valiAUC)
print(f"XG_Boost Validation AUC:{valiAUC[middle_index]:.4f}")
best_model = models[middle_index]

# 在測試集上計算 log_loss 及 AUC
test_log_loss_score = log_loss(y_test, best_model.predict_proba(X_test)[:, 1])
test_auc_score=roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

print(f"XG_Boost Test log_loss: {test_log_loss_score:.4f}")
print(f"XG_Boost Test AUC: {test_auc_score:.4f}")

plot_importance(best_model, max_num_features=10, importance_type='gain', height=0.5)
plt.title("Top 10 Feature Importances by Gain")
plt.tight_layout()
plt.show()