import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, log_loss
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import featuretools as ft
from featuretools.primitives import MultiplyNumeric, DivideNumeric, AddNumeric, SubtractNumeric
import numpy as np
from sklearn.impute import SimpleImputer


# Step 1:載入資料

df = pd.read_csv("./train.csv")
df = df.iloc[:, 1:]
#==========================================================================================================

#Step 2:建立特徵與標籤

# 特徵建構
X = df.iloc[:, :-1]

#建立EntitySet
es = ft.EntitySet(id='smokingdata')
es = es.add_dataframe(dataframe_name="health", dataframe=X, index="index", make_index=True)

#使用 DFS 建立自動特徵(特徵組合)
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="health",
    trans_primitives=[MultiplyNumeric(), DivideNumeric(), AddNumeric(), SubtractNumeric()],
    max_depth=2  # 控制特徵複雜度
)

print("自動產生特徵 shape:", feature_matrix.shape)

# 建立分層特徵（例如把身高分成五等分）
#X["height_level"] = pd.cut(X["height(cm)"], bins=[0, 150, 160, 170, 180, 300],
#                           labels=["very_short", "short", "medium", "tall", "very_tall"])
#X["weight_level"] = pd.cut(X["weight(kg)"], bins=[0, 40, 50, 60, 80, 200],
#                           labels=["thin", "still thin", "medium", "fat", "very fat"])

X=feature_matrix

Y = df["smoking"]

#缺失值處理
X = X.replace([np.inf, -np.inf], np.nan)
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

print("特徵 X 的 shape:", X.shape)
print("標籤 Y 的 shape:", Y.shape)

#==========================================================================================================





#==========================================================================================================

#Step 3:分割訓練與測試資料、特徵降維

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=32, shuffle=True)

"""
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lfda = LFDA(k=5,n_components=1000)
lfda.fit(X_train, y_train)
X_train = lfda.transform(X_train)
X_test = lfda.transform(X_test)
"""
print("特徵 X 的 shape:", X_train.shape)

#==========================================================================================================

#Step 4:建立模型

# 定義 CatBoost 模型
model = CatBoostClassifier(
    iterations=1000,
    depth=5,
    learning_rate=0.05,
    l2_leaf_reg=5,
    cat_features=[],  # 處理所有特徵為數值，若有分類變數，需指定`cat_features`為分類特徵的索引
    random_state=42,
    #verbose=200,
    eval_metric='AUC',
    sampling_frequency='PerTree',
    rsm=0.6
)

#==========================================================================================================

#Step5 調較超參數(訓練過程中以調整完畢，這裡節省時間先註解掉)

# 設定超參數範圍
param_grid = {
    #'iterations': [500, 1000, 1500],
    #'depth': [3,4, 5, 6, 7],
    #'learning_rate': [0.01,0.05],
    #'l2_leaf_reg': [1,2,3,4,5]
}

# 設定 GridSearchCV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='roc_auc',
    refit='roc_auc',  # 選 AUC 最佳的模型
    cv=skf,
    #verbose=1,
    n_jobs=-1
)

# 執行 Grid Search
grid_search.fit(X_train, y_train,early_stopping_rounds=200)

# 最佳模型
print("Best Parameters:", grid_search.best_params_)
print("Best CV AUC:", grid_search.best_score_)

best_model = grid_search.best_estimator_

#==========================================================================================================

#Step 6:檢查測試集表現
 
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

# 轉成類別預測結果(0、1)
y_pred = best_model.predict(X_test)

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred)

# 顯示混淆矩陣
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Smoker", "Smoker"])
disp.plot(cmap=plt.cm.Blues)
plt.title("CatBoost Confusion Matrix")
plt.show()

#==========================================================================================================

#Step 7:預測

# 載入測試資料，進行預測
testcsv = pd.read_csv("./test.csv")
test_feature = testcsv.iloc[:, 1:]

#test_feature["height_level"] = pd.cut(test_feature["height(cm)"], bins=[0, 150, 160, 170, 180, 300],
#                           labels=["very_short", "short", "medium", "tall", "very_tall"])
#test_feature["weight_level"] = pd.cut(test_feature["weight(kg)"], bins=[0, 40, 50, 60, 80, 200],
#                           labels=["thin", "still thin", "medium", "fat", "very fat"])

es_test = ft.EntitySet(id='test_data')
es_test = es_test.add_dataframe(dataframe_name="health", dataframe=test_feature, index="index", make_index=True)

test_feature= ft.calculate_feature_matrix(
    features=feature_defs,
    entityset=es_test
)

print("測試特徵的大小:", test_feature.shape)

#缺失值處理
test_feature = test_feature.replace([np.inf, -np.inf], np.nan)
imputer = SimpleImputer(strategy='median')
test_feature = imputer.fit_transform(test_feature)
"""
#標準化
test_feature = scaler.transform(test_feature)

# 特徵降維
test_feature = lfda.transform(test_feature)
"""
# 預測
answer = best_model.predict(test_feature)

# 產生檔案
submission = pd.DataFrame({
    "id": testcsv.iloc[:, 0],
    "smoking": answer
})
submission.to_csv("answer_catboost_grid.csv", index=False)

print("生成 answer_catboost_grid.csv！")
