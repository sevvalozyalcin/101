# -*- coding: utf-8 -*-
"""20201704073_SevvalOzyalcin_NormalDistribution.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sLyJhplxBOtphBiqTALfc0M-A2Kcjblz
"""

from scipy.stats import kstest, normaltest
float_feats = []
object_feats = []
binary_feats = []
label = []

for col in df.columns:
    if df[col].nunique() == 2:
        binary_feats.append(col)
    elif df[col].nunique() > 30:
        if df[col].dtype == 'float64':
            float_feats.append(col)
        elif df[col].dtype == 'int64':
            binary_feats.append(col)
    else:
        object_feats.append(col)

binary_feats.remove("col122")
label.append("col122")


for col in float_feats:
    stat, p = kstest(df[col], 'norm')
    print(col, "floats p-Test: Statistics={0:.3f}".format(stat, p)) 
    if p < 0.05:
       print(col, "is not normally distributed")
    else:
        print(col, "is normally distributed")

for col in binary_feats:
    stat, p = normaltest(df[col])
    print(col, "integers p-Test: Statistics={0:.3f}".format(stat, p))
    if p < 0.05:
        print(col,"is not normally distributed")
    else:
        print(col, "is normally distributed")
