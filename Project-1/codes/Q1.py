import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt

print("\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")

df =pd.read_csv("../football_data.csv", index_col="Unnamed: 0")
print(df)


print("\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")
# info for all dataset columns_name, dataType, null_count
print(df.info())

print("\n\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")
print(df.shape)


# describe of data min, max, mean, std values for all columns
df.describe(include='all')


club = set(df["Club"].tolist())
dict = {}
dict_new = {}

for i in clubs:
    dict[i] = []
for i in range(len(df.axes[0])):
    zval = df["Wage"].tolist()[i]
    if '€' in zval: zval = zval[1:]
    if 'K' in zval: zval = float(zval[:-1])*1000
    elif 'M' in zval:
        zval = zval[:-1]
        zval = float(zval)*1000000
    dict[df["Club"].tolist()[i]].append(float(zval))
for i in clubs:
    dict_new[i] = np.average(np.array(dict[i]))

dict_new = sorted(dict_new.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)


