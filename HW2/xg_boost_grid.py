import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, log_loss
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import featuretools as ft
from featuretools.primitives import MultiplyNumeric, DivideNumeric, AddNumeric, SubtractNumeric
from sklearn.impute import SimpleImputer

# 載入資料
df = pd.read_csv("HW2/ncku-cs-data-mining-homework-2/train_with_features.csv")
df = df.iloc[:, 1:]

# 特徵與標籤
X = df.iloc[:, :-1]
Y = df["smoking"]
# ---------- 利用 featuretools 建立自動特徵 ----------
# Step 1: 建立 EntitySet
es = ft.EntitySet(id='smoking_data')
es = es.add_dataframe(dataframe_name="health", dataframe=X, index="index", make_index=True)

# Step 2: 使用 Deep Feature Synthesis 建立自動特徵
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="health",
    trans_primitives=[MultiplyNumeric(), DivideNumeric(), AddNumeric(), SubtractNumeric()],
    max_depth=2  # 控制特徵複雜度
)

print("🔧 自動產生特徵 shape:", feature_matrix.shape)
print("🧠 示意特徵:", feature_matrix.columns[:10].tolist())

# 偏態處理
#skewness = X.skew()
#skewed_features = skewness[abs(skewness) > 1].index.tolist()
#print("偏態特徵欄位：", skewed_features)

# 對偏態特徵做 log1p 處理
#X[skewed_features] = X[skewed_features].apply(lambda x: np.log1p(x))
#X["height*hemo"]=X["hemoglobin"]*X["height(cm)"]
#X["height*weight"]=X["weight(kg)"]*X["height(cm)"]
#X=X.drop(["hearing(left)","hearing(right)","relaxation","eyesight(right)","eyesight(left)"],axis=1)
"""
X=df[["height(cm)",
      "hemoglobin",
      "Gtp",
      "triglyceride",
      "dental caries",
      "LDL",
      "BMI",
      "serum creatinine",
      "Cholesterol"]]
"""
X = feature_matrix

print("特徵 X 的 shape:", X.shape)
print("標籤 Y 的 shape:", Y.shape)

# 分割訓練與測試資料
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
imputer = SimpleImputer(strategy="mean")  # 或 "median"
X_train = imputer.fit_transform(X_train)

# 定義 XGBoost 模型
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    min_child_weight=3,
    gamma=0.5,
    subsample=1,
    colsample_bytree=1,
    reg_alpha=0.1,
    reg_lambda=10,
    learning_rate=0.2,
    #use_label_encoder=False, 
    eval_metric='auc', 
    random_state=42
    )

# 超參數範圍
param_grid = {
    #'n_estimators': [100,150,200,250,300,400,500],

    #'max_depth': [3, 4, 5,6,7],
    #'min_child_weight' : [1,2,3,4,5,6,7],

    #'gamma': [0.3, 0.4, 0.5, 0.6,0.7],

    
    #'subsample': [1,1.2,1.3,1.4],
    #'colsample_bytree': [0.9, 1.0,1.1],

    #'reg_alpha': [0,0.01,0.1,1,10],
    #'reg_lambda': [0,0.01,0.1,1.0,10],

    #'learning_rate': [0.1,0.15,0.2]
}

# 設定 GridSearchCV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring={'logloss': 'neg_log_loss', 'auc': 'roc_auc'},
    refit='auc',  # 讓他最後選 AUC 最佳的模型
    cv=skf,
    verbose=1,
    n_jobs=-1
)


# 執行 Grid Search
grid_search.fit(X_train, y_train)

# 最佳模型
print("Best Parameters:", grid_search.best_params_)
print("Best CV log loss:", grid_search.best_score_)

best_model = grid_search.best_estimator_

# 測試集表現
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
test_log_loss_score = log_loss(y_test, y_pred_proba)
test_auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"XGBoost Test log_loss: {test_log_loss_score:.4f}")
print(f"XGBoost Test AUC: {test_auc_score:.4f}")

# 畫出特徵重要性
plot_importance(best_model, max_num_features=31, importance_type='gain', height=0.5)
plt.title("Top 10 Feature Importances by Gain")
plt.tight_layout()
plt.show()

# 轉成類別預測結果（0 or 1）
y_pred = best_model.predict(X_test)

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred)

# 顯示混淆矩陣
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Smoker", "Smoker"])
disp.plot(cmap=plt.cm.Blues)
plt.title("XGBoost Confusion Matrix")
plt.show()

# 載入測試資料，進行預測
testcsv = pd.read_csv("HW2/ncku-cs-data-mining-homework-2/test_with_features.csv")
test_feature = testcsv.iloc[:, 1:]

# 測試資料也要建立 EntitySet 並產生相同特徵
es_test = ft.EntitySet(id='test_data')
es_test = es_test.add_dataframe(dataframe_name="health", dataframe=test_feature, index="index", make_index=True)

test_feature= ft.calculate_feature_matrix(
    features=feature_defs,
    entityset=es_test
)

#test_feature["height*hemo"]=test_feature["hemoglobin"]*test_feature["height(cm)"]
#test_feature["height*weight"]=test_feature["weight(kg)"]*test_feature["height(cm)"]
print("測試特徵的大小:", test_feature.shape)

# 預測
answer = best_model.predict(test_feature)

# 產生提交檔案
submission = pd.DataFrame({
    "id": testcsv.iloc[:, 0],
    "smoking": answer
})
submission.to_csv("HW2/answer_xgboost_grid.csv", index=False)
print("成功生成 answer_xgboost_grid.csv！")
