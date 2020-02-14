import numpy as np
import pandas as pd

file_path = "./"
file_name = "attrition_prediction-Attrition.csv"

raw_data = pd.read_csv(file_path+file_name, index_col=0)
print(raw_data.head)

raw_data = raw_data.drop("Attrition_No_score", axis=1)
print(raw_data)

# raw_data["Attrition_Yes_score"] = raw_data["Attrition_Yes_score"].map(lambda x: 1 if x >= 0.5 else 0)
# print(raw_data["Attrition_Yes_score"])
raw_data = raw_data.rename(columns = {"Attrition_Yes_score": "Attrition"})
print(raw_data.head)
raw_data[['Attrition']].to_csv('submission_automl_raw.csv')