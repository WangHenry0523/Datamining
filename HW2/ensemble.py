import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, log_loss
from catboost import CatBoostClassifier, Pool
from xgboost import plot_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

# 載入資料
df = pd.read_csv("HW2/ncku-cs-data-mining-homework-2/train_with_features.csv")
df = df.iloc[:, 1:]

# 特徵與標籤
X = df.iloc[:, :-1]
Y = df["smoking"]
print("特徵 X 的 shape:", X.shape)
print("標籤 Y 的 shape:", Y.shape)

# 分割訓練與測試資料
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=32, shuffle=True)

# 定義 CatBoost 模型
model = CatBoostClassifier(
    iterations=1000,
    depth=5,
    learning_rate=0.05,
    l2_leaf_reg=5,
    cat_features=[],  # 預設處理所有特徵為數值，若有分類變數，需指定 `cat_features` 為分類特徵的索引
    random_state=42,
    #verbose=200,
    eval_metric='AUC'
)
lightmodel = LGBMClassifier(
    random_state=42,
    n_estimators=150,
    max_depth=4,
    colsample_bytree=0.9,
    subsample=1,
    reg_alpha=0,
    reg_lambda=0
    )
xgbmodel = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    min_child_weight=3,
    gamma=0.5,
    subsample=1,
    colsample_bytree=1,
    reg_alpha=0.1,
    reg_lambda=10,
    learning_rate=0.2,
    #use_label_encoder=False, 
    eval_metric='auc', 
    random_state=42,
    )

# 超參數範圍
param_grid = {
    #'iterations': [500, 1000, 1500],
    #'depth': [3,4, 5, 6, 7],
    #'learning_rate': [0.01,0.05],
    #'l2_leaf_reg': [1,2,3,4,5],
    # 可選: 其他 CatBoost 參數
}

# 設定 GridSearchCV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ensemble_model = VotingClassifier(estimators=[
    ('catboost', model),
    ('xgboost', xgbmodel),
    ('lightgbm', lightmodel)
], voting='soft')  # votin

grid_search = GridSearchCV(
    estimator=ensemble_model,
    param_grid=param_grid,
    scoring='roc_auc',
    refit='roc_auc',
    cv=skf,
    n_jobs=-1
)
# 執行 Grid Search
grid_search.fit(X_train, y_train)

# 最佳模型
print("Best Parameters:", grid_search.best_params_)
print("Best CV AUC:", grid_search.best_score_)

best_model = grid_search.best_estimator_

# 測試集表現
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
test_log_loss_score = log_loss(y_test, y_pred_proba)
test_auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"CatBoost Test log_loss: {test_log_loss_score:.4f}")
print(f"CatBoost Test AUC: {test_auc_score:.4f}")
"""
# 畫出特徵重要性
best_model.get_feature_importance(Pool(X_train, label=y_train,cat_features=["age_group"]))
plot_importance(best_model, max_num_features=20, importance_type='ShapValues', height=0.5)
plt.title("Top 10 Feature Importances by SHAP Values")
plt.tight_layout()
plt.show()
"""

# 轉成類別預測結果（0 or 1）
y_pred = best_model.predict(X_test)

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred)

# 顯示混淆矩陣
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Smoker", "Smoker"])
disp.plot(cmap=plt.cm.Blues)
plt.title("CatBoost Confusion Matrix")
plt.show()

# 載入測試資料，進行預測
testcsv = pd.read_csv("HW2/ncku-cs-data-mining-homework-2/test_with_features.csv")
test_feature = testcsv.iloc[:, 1:]
print("測試特徵的大小:", test_feature.shape)

# 預測
answer = best_model.predict(test_feature)

# 產生提交檔案
submission = pd.DataFrame({
    "id": testcsv.iloc[:, 0],
    "smoking": answer
})
submission.to_csv("HW2/answer_ensemble_grid.csv", index=False)
print("成功生成 answer_catboost_grid.csv！")
