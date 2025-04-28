import pandas as pd
df =pd.read_csv("menu.csv")
df.iloc[:,[1,2,5]]
print(df)