import pandas as pd

test_df = pd.read_csv("HW2/ncku-cs-data-mining-homework-2/test.csv")
"""
test_fe = test_df.copy()
# BMI
test_fe['BMI'] = test_fe['weight(kg)'] / ((test_fe['height(cm)'] / 100) ** 2)

# Waist-to-Height Ratio
test_fe['Waist_to_Height'] = test_fe['waist(cm)'] / test_fe['height(cm)']

# Pulse Pressure
test_fe['Pulse_Pressure'] = test_fe['systolic'] - test_fe['relaxation']

# Eyesight Difference
test_fe['Eyesight_Diff'] = abs(test_fe['eyesight(left)'] - test_fe['eyesight(right)'])

# Hearing Difference
test_fe['Hearing_Diff'] = abs(test_fe['hearing(left)'] - test_fe['hearing(right)'])

# Cholesterol to HDL ratio（避免除以 0）
test_fe['Chol_to_HDL'] = test_fe['Cholesterol'] / test_fe['HDL'].replace(0, 0.1)

# AST to ALT ratio（避免除以 0）
test_fe['AST_to_ALT'] = test_fe['AST'] / test_fe['ALT'].replace(0, 0.1)

# 年齡分群（加上 age_group）
test_fe['age_group'] = pd.cut(test_fe['age'], bins=[0, 30, 45, 60, 150], labels=[0, 1, 2, 3])

test_fe.to_csv("HW2/ncku-cs-data-mining-homework-2/test_with_features.csv", index=False)
"""
