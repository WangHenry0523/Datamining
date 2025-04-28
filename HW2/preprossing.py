import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,KFold
import seaborn as sns
import matplotlib.pyplot as plt

#輸入資料
df=pd.read_csv("HW2/ncku-cs-data-mining-homework-2/train_with_features.csv")
df=df.iloc[:,1:]
#print(df)
"""
#特徵標準化
scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)  #訓練集標準化
X_test= scaler.transform(X_test)  #測試集標準化
"""
#設定特徵與預測結果
X=df.iloc[:,:-1]
X=X.drop(["hearing(left)","hearing(right)","systolic","relaxation"],axis=1)

Y=df["smoking"]
print("特徵 X 的 shape:", X.shape)
print("標籤 Y 的 shape:", Y.shape)

print("====================================================")
#檢查缺失值
#print(df.isnull().sum(axis=0))


print("====================================================")
#查看相關係數、資料分布
# 計算相關係數矩陣
corr_matrix = df.corr()

# 顯示與 "smoking" 的相關係數，從大到小排序
smoking_corr = corr_matrix["smoking"].drop("smoking").sort_values(key=abs, ascending=False)
print("與抽菸習慣最相關的特徵：\n", smoking_corr.head(29))

# 繪製相關係數熱力圖（包含 smoking）
"""
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, fmt=".1f", square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
"""
# 分析抽菸與幾個特徵的關係（以分布圖呈現）
#important_features = smoking_corr.head(10).index.tolist()
important_features=["weight(kg)"]
for feature in important_features:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=df, x=feature, hue="smoking", common_norm=False, fill=True)
    plt.title(f"Distribution of {feature} by Smoking")
    plt.tight_layout()
    plt.show()

#切分資料集
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True)



#設定交叉驗證折數
kf = KFold(n_splits=5,shuffle=True,random_state=42)
