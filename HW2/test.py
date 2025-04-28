from sklearn.feature_selection import VarianceThreshold
import pandas as pd

# 輸入資料
df = pd.read_csv("HW2/ncku-cs-data-mining-homework-2/train.csv")
df = df.iloc[:, 1:]  # 假設第一列是ID或不需要的資料，進行去除

# 設定特徵與預測結果
X = df.iloc[:, :-1]  # 所有欄位，除了最後一個作為特徵
Y = df["smoking"]  # 標籤 Y

# 顯示特徵與標籤的形狀
print("特徵 X 的 shape:", X.shape)
print("標籤 Y 的 shape:", Y.shape)

# 計算每個特徵的變異數
variances = X.var()
print("每個特徵的變異數：")
print(variances)

# 使用 VarianceThreshold 剔除變異數小於 0.1 的特徵
selector = VarianceThreshold(threshold=0.1)  # 設為 0.1 選擇變異數大於 0.1 的特徵
X_selected = selector.fit_transform(X)  # 拟合並變換資料

# 取得被選中的特徵
selected_features = X.columns[selector.get_support()]
print(f"被選中的特徵：{selected_features}")

# 更新 X 變數以包含選中的特徵
X = X[selected_features]
print("特徵 X 的 shape:", X.shape)
