import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt

# 載入資料
df = pd.read_csv("HW2/ncku-cs-data-mining-homework-2/train.csv")
df = df.iloc[:, 1:]

# 特徵與標籤
X = df.iloc[:, :-1]
Y = df["smoking"]

print("特徵 X 的 shape:", X.shape)
print("標籤 Y 的 shape:", Y.shape)

# 資料切分：先分出訓練 + 測試
X_train_full, X_test, y_train_full, y_test = train_test_split(X, Y, test_size=0.2, random_state=22, shuffle=True)

# 再從訓練集中切出驗證集（Holdout 法）
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=22)

# 模型訓練
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# 評估
train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
test_logloss = log_loss(y_test, model.predict_proba(X_test)[:, 1])

print(f"Train AUC: {train_auc:.4f}")
print(f"Validation AUC: {val_auc:.4f}")
print(f"XG_Boost Test AUC: {test_auc:.4f}")
print(f"XG_Boost Test log_loss: {test_logloss:.4f}")

# 特徵重要性
plot_importance(model, max_num_features=10, importance_type='gain', height=0.5)
plt.title("Top 10 Feature Importances by Gain")
plt.tight_layout()
plt.show()

# 預測測試集
testcsv = pd.read_csv("HW2/ncku-cs-data-mining-homework-2/test.csv")
test_feature = testcsv.iloc[:, 1:]
print("測試特徵的大小", test_feature.shape)

answer = model.predict(test_feature)

# 生成提交檔案
submission = pd.DataFrame({
    "id": testcsv.iloc[:, 0],
    "smoking": answer
})
submission.to_csv("HW2/answer_xgboost_holdout.csv", index=False)
print("成功生成 answer_xgboost_holdout.csv！")
