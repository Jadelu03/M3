# -*- coding: utf-8 -*-

#imports
from sys import exit
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import zscore


PROJ = 'M3s'
cdoc = 'M3s_wine_Chem.csv'
xdoc = 'M3s_wine_NMR.csv'
cat = 'color'

#dirs
cwd = os.getcwd()
cwd = cwd.replace("\\", "/")
projdir = cwd + '/Projects/' +  PROJ + '/'
resultsdir = projdir + 'PLS/'

#make work folders
folds = [projdir,resultsdir]
for fold in folds:
    if not os.path.exists(fold):
        os.makedirs(fold)

# load c data
fil = projdir + cdoc
print(fil)
cdf = pd.read_csv(fil, sep = ",")

# load x data
fil = projdir + xdoc
print(fil)
xdf = pd.read_csv(fil, sep = ",")

# Drop rows with any NaN values by index
xdrops = xdf[xdf.isna().any(axis=1)].index.tolist()
cdrops = cdf[cdf.isna().any(axis=1)].index.tolist()
nans = xdrops + cdrops
print(nans)

cdf = cdf.drop(nans)
xdf = xdf.drop(nans) 

#get colors and ids
levs = xdf['color'].tolist()
myid = xdf['Sample'].tolist() #for labelling by index if needed

#drop unwanted columns
columns_to_drop = ['Sample','color']
y = cdf.drop(columns=columns_to_drop, axis=1)
X = xdf.drop(columns=columns_to_drop, axis=1)

#get names
cnames = y.columns.tolist()
xnames = X.columns.tolist()

#simple PLS
n_comp = X.shape[1]
pls = PLSRegression(n_components=n_comp, scale=True)
pls.fit(X, y)

#loadings and scores
t_pls = pls.x_scores_
w_pls = pls.x_weights_
q_pls = pls.y_loadings_

#colors for wines samples
colordic = {'red':'red','rose':'pink','white':'yellow'}

#plot
plt.figure(figsize=(8,6))

#scaling for biplot
wscale = 100
qscale = 100
sscale = 1

#w, weighted P loadings (explanitory)
for i in range(len(w_pls)):
    plt.scatter(w_pls[i][0]*wscale,w_pls[i][1]*wscale, s= 50, alpha = 0.8, marker='^',edgecolor='black')
    plt.text(w_pls[i][0]*wscale,w_pls[i][1]*wscale, xnames[i],fontsize=7,color='black',alpha = 0.8)

#q, weighted C loadings (response)
for i in range(len(q_pls)):
    plt.scatter(q_pls[i][0]*qscale,q_pls[i][1]*qscale, s= 400, alpha = 0.8,marker='s',edgecolor='black')
    plt.text(q_pls[i][0]*qscale,q_pls[i][1]*qscale, cnames[i],fontsize=13,color='blue')
 
#samples (T)
used = []
scores = t_pls
for i in range(len(scores)):
    plt.text(scores[i][0]*sscale,scores[i][1]*sscale, myid[i],fontsize=10,color='green')
    if levs[i] in used:
        plt.scatter(scores[i][0]*sscale,scores[i][1]*sscale, s=100, alpha = 0.6, color = colordic[levs[i]],edgecolor='black')
    else:
        plt.scatter(scores[i][0]*sscale,scores[i][1]*sscale, s=100, label=levs[i], color = colordic[levs[i]], alpha = 0.6,edgecolor='black')
        used.append(levs[i])

#lines
plt.axhline(0, color='k', linestyle='-',linewidth=0.7)
plt.axvline(0, color='k', linestyle='-',linewidth=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')

#legend
plt.legend(title=cat, loc=4, fontsize = 'small')
tit = PROJ + ', PLS Plot'
plt.title(tit, fontsize = 16)

#save
label = resultsdir + PROJ + '_PLS_biplot_' + cat + '.png'
plt.savefig(label, format='png', dpi=1200, bbox_inches='tight')
plt.show()
plt.close()