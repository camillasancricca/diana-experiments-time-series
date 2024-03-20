from Lib.injection.dirty_accuracy_rows import injection
import pandas as pd
import pickle

path = "../../Datasets/PRSA_Data_imputed.csv"

df = pd.read_csv(path, sep=",")

#df = df.reset_index(drop=True)
_, outliers_mask = injection(df, 1, "PRSA_Data", name_class=["PM2.5", "station"])
print(len(outliers_mask))
with open('../../Datasets/outliers_index.pkl', 'wb') as pick:
    pickle.dump(outliers_mask,pick)


