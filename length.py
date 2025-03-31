import xlrd

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

xls = pd.ExcelFile ('excel/dataset_nuovo_short.xlsx')

df1 = pd.read_excel(xls, 'VAW')
df2 = pd.read_excel(xls, 'Ar')
df3 = pd.read_excel(xls, 'A')

data_set = pd.concat([df1, df2, df3], ignore_index=True)

df = data_set[['length', 'label']]

bins = [1, 10, 20, 30, 40,50,60,70,80,90,100, 200, 300, 400, 500, 600, 700, 800]

labels = ["1-10", "10-20", "20-30","30-40","40-50", "50-60","60-70","70-80","80-90","90-100", "100-200", "200-300", "300-400", "400-500", "500-600", "600-700", "700-800"]


groups = df.groupby(pd.cut(df.length, bins=bins, labels=labels))
result = groups.length.count().reset_index(name="count")

print(result)

ax = result.plot.bar(x='length', y='count', rot=0)

plt.show()