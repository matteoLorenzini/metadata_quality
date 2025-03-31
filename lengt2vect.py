import pandas as pd

df = pd.read_csv ('CH_IT/CSV/A/AFT.csv', names=['DESCRIPTION','LABEL'])

df = pd.DataFrame(df)

df['LENGTH']  = df['DESCRIPTION'].str.len()

df.loc[df['LABEL'] == 'Good', 'label_match'] = '1'  
df.loc[df['LABEL'] != 'Good', 'label_match'] = '0'

df2 = df[['label_match', 'LENGTH']]


df2['vect'] = df2['LENGTH'].apply(lambda x: "{}{}".format('1:', x))

df3 = df2[['label_match','vect']]

print(df3)


df3.to_csv('A_lenght.tsv', sep = '\t', index=False, header=False)


#VAW

df4 = pd.read_csv ('CH_IT/CSV/vaw/vawFT.csv', names=['DESCRIPTION','LABEL'])

df4 = pd.DataFrame(df4)

df4['LENGTH']  = df4['DESCRIPTION'].str.len()

df4.loc[df4['LABEL'] == 'Good', 'label_match'] = '1'  
df4.loc[df4['LABEL'] != 'Good', 'label_match'] = '0'

df5 = df4[['label_match', 'LENGTH']]


df5['vect'] = df5['LENGTH'].apply(lambda x: "{}{}".format('1:', x))

df6 = df5[['label_match','vect']]

print(df6)


df6.to_csv('vaw_lenght.tsv', sep = '\t', index=False, header=False)

#Ar

df7 = pd.read_csv ('CH_IT/CSV/Ar/ArFT.csv', names=['DESCRIPTION','LABEL'])

df7 = pd.DataFrame(df7)

df7['LENGTH']  = df7['DESCRIPTION'].str.len()

df7.loc[df7['LABEL'] == 'Good', 'label_match'] = '1'  
df7.loc[df7['LABEL'] != 'Good', 'label_match'] = '0'

df8 = df7[['label_match', 'LENGTH']]


df8['vect'] = df8['LENGTH'].apply(lambda x: "{}{}".format('1:', x))

df9 = df8[['label_match','vect']]

print(df9)


df9.to_csv('ar_lenght.tsv', sep = '\t', index=False, header=False)

#total

df10 = pd.read_csv ('CH_IT/CSV/total/totalFT.csv', names=['DESCRIPTION','LABEL'])

df10 = pd.DataFrame(df10)

df10['LENGTH']  = df10['DESCRIPTION'].str.len()

df10.loc[df10['LABEL'] == 'Good', 'label_match'] = '1'  
df10.loc[df10['LABEL'] != 'Good', 'label_match'] = '0'

df11 = df10[['label_match', 'LENGTH']]


df11['vect'] = df11['LENGTH'].apply(lambda x: "{}{}".format('1:', x))

df12 = df11[['label_match','vect']]

print(df12)


df12.to_csv('total_lenght.tsv', sep = '\t', index=False, header=False)