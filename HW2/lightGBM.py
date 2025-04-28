"""
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import roc_auc_score,log_loss
from lightgbm import LGBMClassifier, plot_importance
import matplotlib.pyplot as plt

#輸入資料
df=pd.read_csv("HW2/ncku-cs-data-mining-homework-2/train_with_features.csv")
df=df.iloc[:,1:]
#print(df)

#設定特徵與預測結果
X=df.iloc[:,:-1]
Y=df["smoking"]

print("特徵 X 的 shape:", X.shape)
print("標籤 Y 的 shape:", Y.shape)

#切分資料集
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,random_state=32,shuffle=True)

#設定交叉驗證折數
kf = KFold(n_splits=5,shuffle=True,random_state=32)

#Logistic Regression

trainAUC=[]
valiAUC=[]
models=[]

for trainIndex,valIndex in kf.split(X_train):
    X_fold_train,X_fold_val = X_train.iloc[trainIndex], X_train.iloc[valIndex]  #根據折數切出每一份特徵
    y_fold_train,y_fold_val = y_train.iloc[trainIndex], y_train.iloc[valIndex]  #根據折數切出每一份預測結果
    
    model = LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )
    model.fit(X_fold_train,y_fold_train)    #訓練模型
    
    # 計算loss
    train_auc=roc_auc_score(y_fold_train, model.predict_proba(X_fold_train)[:, 1])
    val_auc= roc_auc_score(y_fold_val, model.predict_proba(X_fold_val)[:, 1])
    # 填入loss串列
    trainAUC.append(train_auc)
    valiAUC.append(val_auc)
    #填入models串列
    models.append(model)

# 找到最中間的那一折
middle_index = np.argmax(valiAUC)
print(f"lightGBM Validation AUC:{valiAUC[middle_index]:.4f}")
best_model = models[middle_index]

# 在測試集上計算 log_loss
test_log_loss_score = log_loss(y_test, best_model.predict_proba(X_test)[:, 1])
test_auc_score=roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

# 顯示結果
#print(f"Train log_loss (每折): {trainLoss}")
#print(f"Validation log_loss (每折): {valiLoss}")
print(f"lightGBM Test log_loss: {test_log_loss_score:.4f}")
print(f"lightGBM Test AUC: {test_auc_score:.4f}")

plot_importance(best_model, max_num_features=10, importance_type='gain', height=0.5)
plt.title("Top 20 Feature Importances by Gain")
plt.tight_layout()
plt.show()

#寫入預測結果
testcsv=pd.read_csv("HW2/ncku-cs-data-mining-homework-2/test_with_features.csv")

test_feature=testcsv.iloc[:,1:]
print("測試特徵的大小",test_feature.shape)
#test_feature = scaler.transform(test_feature)
answer=best_model.predict(test_feature)

# 生成提交格式（假設 ID 是 test.csv 裡第一欄）
submission = pd.DataFrame({
    "id": testcsv.iloc[:,0], # 假設第一欄是 id
    "smoking": answer        # 預測的機率
})

# 寫入 CSV（不寫 index）
submission.to_csv("HW2/answer_lightgbm.csv", index=False)
print("成功生成 answer.csv！")
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import roc_auc_score, log_loss
from lightgbm import LGBMClassifier, plot_importance
import matplotlib.pyplot as plt

# 輸入資料
df = pd.read_csv("HW2/ncku-cs-data-mining-homework-2/train_with_features.csv")
df = df.iloc[:, 1:]

# 設定特徵與預測結果
X = df.iloc[:, :-1]
Y = df["smoking"]

print("特徵 X 的 shape:", X.shape)
print("標籤 Y 的 shape:", Y.shape)

# 切分資料集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=32, shuffle=True)

# 設定交叉驗證折數
kf = KFold(n_splits=5, shuffle=True, random_state=32)

# 設定 LGBM 模型
model = LGBMClassifier(
    random_state=42,
    n_estimators=150,
    max_depth=4,
    colsample_bytree=0.9,
    subsample=1,
    reg_alpha=0,
    reg_lambda=0

    )

# 設定 GridSearchCV 參數
param_grid = {
    #'n_estimators': [100,150,200,250,300,350,400,450],
    #'max_depth': [3,4,5,6,7,8],
    'learning_rate': [0.05, 0.1, 0.2],
    #'subsample': [1,1.2,1.3,1.4],
    #'colsample_bytree': [0.9, 1.0,1.1],
    #'reg_alpha': [0,0.01,0.1,1,10],
    #'reg_lambda': [0,0.01,0.1,1.0,10]
    
}

# 使用 GridSearchCV 進行超參數調整
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='roc_auc', n_jobs=-1, verbose=1)

# 執行 GridSearchCV
grid_search.fit(X_train, y_train)

# 顯示最佳參數
print("Best Parameters found: ", grid_search.best_params_)

# 使用最佳參數訓練模型
best_model = grid_search.best_estimator_

# 在測試集上計算 log_loss 和 AUC
test_log_loss_score = log_loss(y_test, best_model.predict_proba(X_test)[:, 1])
test_auc_score = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

# 顯示結果
print(f"lightGBM Test log_loss: {test_log_loss_score:.4f}")
print(f"lightGBM Test AUC: {test_auc_score:.4f}")

# 畫出特徵重要性
plot_importance(best_model, max_num_features=10, importance_type='gain', height=0.5)
plt.title("Top 10 Feature Importances by Gain")
plt.tight_layout()
plt.show()

# 載入測試集進行預測
testcsv = pd.read_csv("HW2/ncku-cs-data-mining-homework-2/test_with_features.csv")
test_feature = testcsv.iloc[:, 1:]

# 預測結果
answer = best_model.predict(test_feature)

# 生成提交格式（假設 ID 是 test.csv 裡第一欄）
submission = pd.DataFrame({
    "id": testcsv.iloc[:, 0],  # 假設第一欄是 id
    "smoking": answer          # 預測的機率
})

# 寫入 CSV（不寫 index）
submission.to_csv("HW2/answer_lightgbm_grid.csv", index=False)
print("成功生成 answer_lightgbm_grid.csv！")
