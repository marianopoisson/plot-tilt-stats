!git clone https://github.com/marianopoisson/ModelingArs ARSFIT

# --- Nueva celda ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
from scipy.io import readsav
from scipy import ndimage
from scipy.stats import skew
import matplotlib.transforms as mtransforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
# import math
import glob as glob
import seaborn as sns
from scipy.stats import linregress


#import sys
#sys.path.append('./ARSFIT/')

from scipy import stats

import os

#from funciones.opencube2 import opencube2
#from funciones.func4 import tilt,elong,MFLUX,barys,set_ranges,sizes
#from funciones.modelcube import modelmag,modelmagf,modelmag2

from IPython.display import display, HTML
display(HTML("<style>.output_scroll {height: 400px; overflow-y: scroll;}</style>"))

# --- Nueva celda ---

%cd /content/ARSFIT/stats-tilt/

# --- Nueva celda ---

DF=pd.read_csv('compare-params-TM3-B.csv')

# --- Nueva celda ---

DF[DF.AR == 8913]=DF[DF.AR == 8913].assign(fint=DF[DF.AR == 8913].flux/np.max(DF[DF.AR == 8913].flux))

# --- Nueva celda ---

DFt=DF.groupby(['AR']).min().reset_index()
len(DFt[(DFt.fint>=0.2) ].AR)

# --- Nueva celda ---

DF.groupby(['AR']).min().reset_index().fint

# --- Nueva celda ---

DF['alpha']=DF.apply(lambda x: -1*180*np.sign(x.lat)*x.alpha/np.pi,axis=1)
DF['alphab']=DF.apply(lambda x: -1*np.sign(x.lat)*x.alphab,axis=1)

# --- Nueva celda ---

DF=DF.assign(rot= lambda x: np.gradient(x.alpha,x.mag)/1.5)
DF=DF.assign(rotb= lambda x: np.gradient(x.alphab,x.mag)/1.5)

# --- Nueva celda ---

lidf=[]

lims=[0,0.25,0.5,0.75,1]

for i in range(4):
    DF2=DF[(DF.fint > lims[i]) & (DF.fint <= lims[i+1])].groupby(['AR']).mean().reset_index()
    DF2=DF2.assign(frange=i+1)

    lidf.append(DF2)

DF2=pd.concat(lidf)




# --- Nueva celda ---



fig, axs = plt.subplots(2, 2, figsize=(10, 7),sharey=True,sharex=True)

axs=np.ravel(axs)


for d in [1,2,3,4]:


    DFx=DF2[DF2['frange']==d]

    sns.histplot(data=DFx,x='alpha',alpha=0.5,bins=10,binrange=(-60,60),label='Bayesian',ax=axs[d-1])

    sns.histplot(data=DFx,x='alphab',alpha=0.5,bins=10,binrange=(-60,60),label='Barycenters',ax=axs[d-1])

   # sns.histplot(data=DFx,x='ratio',alpha=0.5,bins=10,binrange=(-60,60),label='Model-Barycenters')



    axs[d-1].axvline(0,color='black',linestyle='dashed')

    axs[d-1].axvline(DFx.alpha.median(),color='tab:blue')
    axs[d-1].axvline(DFx.alphab.median(),color='tab:orange')
    axs[d-1].text(0.05, 0.75, f"                     \n                 ",
             transform=axs[d-1].transAxes,
             fontsize=12,
             bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round',alpha=0.2))

    mayores_cero=[]
    menores_cero=[]
    for en,at in enumerate(['alpha','alphab']):
        # Calcular los valores mayores y menores que cero
        mayores_cero.append((DFx[at] > 0).sum())
        menores_cero.append((DFx[at] < 0).sum())

    axs[d-1].text(0.05, 0.75, f"α > 0:\nα < 0:", transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.18, 0.75, f"{mayores_cero[0]} \n{menores_cero[0]}", color='tab:blue', transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.24, 0.75, f"{mayores_cero[1]} \n{menores_cero[1]}", color='tab:orange', transform=axs[d-1].transAxes, fontsize=12)


    axs[d-1].text(0.7, 0.58, f"                  \n               ",
         transform=axs[d-1].transAxes,
         fontsize=14,
         bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round',alpha=0.2))

    axs[d-1].text(0.7, 0.65, r"$\alpha_{med}:$", transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.81, 0.65, f"{DFx.alpha.median():.1f}", color='tab:blue', transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.89, 0.65, f"{DFx.alphab.median():.1f}", color='tab:orange', transform=axs[d-1].transAxes, fontsize=12)

    axs[d-1].text(0.7, 0.59, r"$\sigma_\alpha:$", transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.81, 0.59, f"{DFx.alpha.std():.0f}", color='tab:blue', transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.89, 0.59, f"{DFx.alphab.std():.0f}", color='tab:orange', transform=axs[d-1].transAxes, fontsize=12)

       # axs[d-1].text(0.05, 0.80, f"α < 0: ,{menores_cero[1]}", transform=axs[d-1].transAxes, fontsize=12)

        # Agregar texto con la información
        # Agregar texto con formato LaTeX
#    axs[d-1].text(
#    0.05, 0.85-en/5,
 #   rf"$\alpha > 0: \color{{blue}}{{mayores_cero[0]}},{mayores_cero[1]}$"+f"\n"+
 #   rf"$\alpha < 0: {menores_cero[0]},{menores_cero[1]}$",
 #   transform=axs[d-1].transAxes,
 #   fontsize=12,




    axs[d-1].legend()

axs[0].set_ylabel('ARs',fontsize=14)
axs[2].set_ylabel('ARs',fontsize=14)

axs[2].set_xlabel(r'$\alpha$ [deg]',fontsize=14)
axs[3].set_xlabel(r'$\alpha$ [deg]',fontsize=14)

for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=12)



fig.tight_layout(pad=1.0)

plt.savefig('./plotilt/alphas.pdf',dpi=300)

# --- Nueva celda ---

fig, axs = plt.subplots(3, 1, figsize=(6, 12),sharex=True)



for en,ra in enumerate([(0,12),(12,20),(20,45)]):
    axs[en].set_ylim(-50,50)

    axs[en].axhline(0,color='black')

    sns.scatterplot(data=DF[(DF.lat.abs() >=ra[0]) & (DF.lat.abs() <ra[1])],ax=axs[en],x='fint',y='alpha',alpha=0.5,s=30)
    sns.scatterplot(data=DF[(DF.lat.abs() >=ra[0]) & (DF.lat.abs() <ra[1])],ax=axs[en],x='fint',y='alphab',alpha=0.5,s=30)

    sns.regplot(data=DF[(DF.lat.abs() >=ra[0]) & (DF.lat.abs() <ra[1])],ax=axs[en],x='fint',y='alpha',color='tab:red',x_bins=10,label='Bayesian')
    sns.regplot(data=DF[(DF.lat.abs() >=ra[0]) & (DF.lat.abs() <ra[1])],ax=axs[en],x='fint',y='alphab',color='tab:green',x_bins=10,label='Barycenters')
    axs[en].legend()

    axs[en].text(.7,.1,rf'${ra[0]}^\circ<|lat|\leq {ra[1]}^\circ$', transform=axs[en].transAxes, fontsize=12,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round',alpha=0.5))
#    plt.set_xlabel('Normalized flux')
    axs[en].set_ylabel(r'$\alpha$ [deg]')

fig.tight_layout(pad=1.0)

axs[2].set_xlabel('Normalized flux')



# --- Nueva celda ---



fig, axs = plt.subplots(2, 2, figsize=(10, 7),sharey=True,sharex=True)

axs=np.ravel(axs)

DF2x=DF2[(DF2.lat.abs() >=12) & (DF2.lat.abs() <20)]


for d in [1,2,3,4]:


    DFx=DF2x[(DF2x['frange']==d)]

    sns.histplot(data=DFx,x='alpha',alpha=0.5,bins=10,binrange=(-60,60),label='Bayesian',ax=axs[d-1])

    sns.histplot(data=DFx,x='alphab',alpha=0.5,bins=10,binrange=(-60,60),label='Barycenters',ax=axs[d-1])

   # sns.histplot(data=DFx,x='ratio',alpha=0.5,bins=10,binrange=(-60,60),label='Model-Barycenters')



    axs[d-1].axvline(0,color='black',linestyle='dashed')

    axs[d-1].axvline(DFx.alpha.median(),color='tab:blue')
    axs[d-1].axvline(DFx.alphab.median(),color='tab:orange')
    axs[d-1].text(0.05, 0.75, f"                     \n                 ",
             transform=axs[d-1].transAxes,
             fontsize=12,
             bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round',alpha=0.2))

    mayores_cero=[]
    menores_cero=[]
    for en,at in enumerate(['alpha','alphab']):
        # Calcular los valores mayores y menores que cero
        mayores_cero.append((DFx[at] > 0).sum())
        menores_cero.append((DFx[at] < 0).sum())

    axs[d-1].text(0.05, 0.75, f"α > 0:\nα < 0:", transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.18, 0.75, f"{mayores_cero[0]} \n{menores_cero[0]}", color='tab:blue', transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.24, 0.75, f"{mayores_cero[1]} \n{menores_cero[1]}", color='tab:orange', transform=axs[d-1].transAxes, fontsize=12)


    axs[d-1].text(0.75, 0.65, f"                ",
         transform=axs[d-1].transAxes,
         fontsize=12,
         bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round',alpha=0.2))

    axs[d-1].text(0.75, 0.65, r"$\bar{\alpha}:$", transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.8, 0.65, f"{DFx.alpha.median():.1f}", color='tab:blue', transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.87, 0.65, f"{DFx.alphab.median():.1f}", color='tab:orange', transform=axs[d-1].transAxes, fontsize=12)

       # axs[d-1].text(0.05, 0.80, f"α < 0: ,{menores_cero[1]}", transform=axs[d-1].transAxes, fontsize=12)

        # Agregar texto con la información
        # Agregar texto con formato LaTeX
#    axs[d-1].text(
#    0.05, 0.85-en/5,
 #   rf"$\alpha > 0: \color{{blue}}{{mayores_cero[0]}},{mayores_cero[1]}$"+f"\n"+
 #   rf"$\alpha < 0: {menores_cero[0]},{menores_cero[1]}$",
 #   transform=axs[d-1].transAxes,
 #   fontsize=12,




    axs[d-1].legend()

axs[0].set_ylabel('Active regions')
axs[2].set_ylabel('Active regions')

axs[2].set_xlabel(r'$\alpha$ [deg]')
axs[3].set_xlabel(r'$\alpha$ [deg]')


fig.tight_layout(pad=1.0)

fig.show()

# --- Nueva celda ---



# --- Nueva celda ---



fig, axs = plt.subplots(2, 2, figsize=(10, 7),sharey=True,sharex=True)

axs=np.ravel(axs)


for d in [1,2,3,4]:


    DFx=DF2[DF2['frange']==d]

    sns.histplot(data=DFx,x='rot',alpha=0.5,bins=10,binrange=(-2,2),label='Bayesian',ax=axs[d-1])

    sns.histplot(data=DFx,x='rotb',alpha=0.5,bins=10,binrange=(-2,2),label='Barycenters',ax=axs[d-1])

   # sns.histplot(data=DFx,x='ratio',alpha=0.5,bins=10,binrange=(-60,60),label='Model-Barycenters')



    axs[d-1].axvline(0,color='black',linestyle='dashed')

    axs[d-1].axvline(DFx.rot.median(),color='tab:blue')
    axs[d-1].axvline(DFx.rotb.median(),color='tab:orange')
    axs[d-1].text(0.05, 0.75, f"                     \n                 ",
             transform=axs[d-1].transAxes,
             fontsize=12,
             bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round',alpha=0.2))

    mayores_cero=[]
    menores_cero=[]
    for en,at in enumerate(['rot','rotb']):
        # Calcular los valores mayores y menores que cero
        mayores_cero.append((DFx[at] > 0).sum())
        menores_cero.append((DFx[at] < 0).sum())

    axs[d-1].text(0.05, 0.75, f"Δα > 0:\nΔα < 0:", transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.2, 0.75, f"{mayores_cero[0]} \n{menores_cero[0]}", color='tab:blue', transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.26, 0.75, f"{mayores_cero[1]} \n{menores_cero[1]}", color='tab:orange', transform=axs[d-1].transAxes, fontsize=12)


    axs[d-1].text(0.7, 0.58, f"                    \n               ",
         transform=axs[d-1].transAxes,
         fontsize=14,
         bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round',alpha=0.2))

    axs[d-1].text(0.7, 0.65, r"$\Delta\alpha_{med}:$", transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.83, 0.65, f"{DFx.rot.median():.1f}", color='tab:blue', transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.91, 0.65, f"{DFx.rotb.median():.1f}", color='tab:orange', transform=axs[d-1].transAxes, fontsize=12)

    axs[d-1].text(0.7, 0.59, r"$\sigma_{\Delta\alpha}:$", transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.83, 0.59, f"{DFx.rot.std():.1f}", color='tab:blue', transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.91, 0.59, f"{DFx.rotb.std():.1f}", color='tab:orange', transform=axs[d-1].transAxes, fontsize=12)

       # axs[d-1].text(0.05, 0.80, f"α < 0: ,{menores_cero[1]}", transform=axs[d-1].transAxes, fontsize=12)

        # Agregar texto con la información
        # Agregar texto con formato LaTeX
#    axs[d-1].text(
#    0.05, 0.85-en/5,
 #   rf"$\alpha > 0: \color{{blue}}{{mayores_cero[0]}},{mayores_cero[1]}$"+f"\n"+
 #   rf"$\alpha < 0: {menores_cero[0]},{menores_cero[1]}$",
 #   transform=axs[d-1].transAxes,
 #   fontsize=12,




    axs[d-1].legend()

axs[0].set_ylabel('Active regions',fontsize=14)
axs[2].set_ylabel('Active regions',fontsize=14)

axs[2].set_xlabel(r'$\Delta \alpha$ [deg/hour]',fontsize=14)
axs[3].set_xlabel(r'$\Delta \alpha$ [deg/hour]',fontsize=14)

for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=12)


fig.tight_layout(pad=1.0)

plt.savefig('./plotilt/rotalpha.pdf',dpi=300)

# --- Nueva celda ---



fig, axs = plt.subplots(2, 2, figsize=(10, 7),sharey=True,sharex=True)

axs=np.ravel(axs)


for d in [1,2,3,4]:


    DFx=DF2[DF2['frange']==d]
    DFx=DFx.assign(ratea=lambda x: x.alpha*x.rot)
    DFx=DFx.assign(rateb=lambda x: x.alphab*x.rotb)

    DFx=DFx[(DFx.ratea.abs()<=10) & (DFx.rateb.abs()<=10)]

    sns.histplot(data=DFx,x='ratea',alpha=0.5,bins=10,binrange=(-8,8),label='Bayesian',ax=axs[d-1])

    sns.histplot(data=DFx,x='rateb',alpha=0.5,bins=10,binrange=(-8,8),label='Barycenters',ax=axs[d-1])

   # sns.histplot(data=DFx,x='ratio',alpha=0.5,bins=10,binrange=(-60,60),label='Model-Barycenters')



    axs[d-1].axvline(0,color='black',linestyle='dashed')

    axs[d-1].axvline(DFx.ratea.mean(),color='tab:blue')
    axs[d-1].axvline(DFx.rateb.mean(),color='tab:orange')
    axs[d-1].text(0.05, 0.75, f"                     \n                 ",
             transform=axs[d-1].transAxes,
             fontsize=12,
             bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round',alpha=0.2))

    mayores_cero=[]
    menores_cero=[]
    for en,at in enumerate(['ratea','rateb']):
        # Calcular los valores mayores y menores que cero
        mayores_cero.append((DFx[at] > 0).sum())
        menores_cero.append((DFx[at] < 0).sum())

    axs[d-1].text(0.05, 0.75, f"αΔα > 0:\nαΔα < 0:", transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.2, 0.75, f"{mayores_cero[0]} \n{menores_cero[0]}", color='tab:blue', transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.26, 0.75, f"{mayores_cero[1]} \n{menores_cero[1]}", color='tab:orange', transform=axs[d-1].transAxes, fontsize=12)


    axs[d-1].text(0.7, 0.58, f"                    \n               ",
         transform=axs[d-1].transAxes,
         fontsize=14,
         bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round',alpha=0.2))

    axs[d-1].text(0.7, 0.65, r"$\Delta\alpha_{med}:$", transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.83, 0.65, f"{DFx.ratea.median():.1f}", color='tab:blue', transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.91, 0.65, f"{DFx.rateb.median():.1f}", color='tab:orange', transform=axs[d-1].transAxes, fontsize=12)

    axs[d-1].text(0.7, 0.59, r"$\sigma_{\Delta\alpha}:$", transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.83, 0.59, f"{DFx.ratea.std():.1f}", color='tab:blue', transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.91, 0.59, f"{DFx.rateb.std():.1f}", color='tab:orange', transform=axs[d-1].transAxes, fontsize=12)

       # axs[d-1].text(0.05, 0.80, f"α < 0: ,{menores_cero[1]}", transform=axs[d-1].transAxes, fontsize=12)

        # Agregar texto con la información
        # Agregar texto con formato LaTeX
#    axs[d-1].text(
#    0.05, 0.85-en/5,
 #   rf"$\alpha > 0: \color{{blue}}{{mayores_cero[0]}},{mayores_cero[1]}$"+f"\n"+
 #   rf"$\alpha < 0: {menores_cero[0]},{menores_cero[1]}$",
 #   transform=axs[d-1].transAxes,
 #   fontsize=12,




    axs[d-1].legend()

axs[0].set_ylabel('ARs',fontsize=14)
axs[2].set_ylabel('ARs',fontsize=14)

axs[2].set_xlabel(r'$\alpha\Delta \alpha$ [deg**2/hour]',fontsize=14)
axs[3].set_xlabel(r'$\alpha\Delta \alpha$ [deg**2/hour]',fontsize=14)

for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=12)

fig.tight_layout(pad=1.0)

# --- Nueva celda ---



fig, axs = plt.subplots(2, 2, figsize=(10, 7),sharey=True,sharex=True)

axs=np.ravel(axs)


for d in [1,2,3,4]:


    DFx=DF2[DF2['frange']==d]
    DFx=DFx.assign(ratea=lambda x: x.rot/x.alpha)
    DFx=DFx.assign(rateb=lambda x: x.rotb/x.alphab)

    DFx=DFx[(DFx.ratea.abs()<=1) & (DFx.rateb.abs()<=1)]

    sns.histplot(data=DFx,x='ratea',alpha=0.5,bins=20,binrange=(-0.5,0.5),label='Bayesian',ax=axs[d-1])

    sns.histplot(data=DFx,x='rateb',alpha=0.5,bins=20,binrange=(-0.5,0.5),label='Barycenters',ax=axs[d-1])

   # sns.histplot(data=DFx,x='ratio',alpha=0.5,bins=10,binrange=(-60,60),label='Model-Barycenters')



    axs[d-1].axvline(0,color='black',linestyle='dashed')

    axs[d-1].axvline(DFx.ratea.median(),color='tab:blue')
    axs[d-1].axvline(DFx.rateb.median(),color='tab:orange')
    axs[d-1].text(0.05, 0.75, f"                           \n                 ",
             transform=axs[d-1].transAxes,
             fontsize=12,
             bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round',alpha=0.2))

    mayores_cero=[]
    menores_cero=[]
    for en,at in enumerate(['ratea','rateb']):
        # Calcular los valores mayores y menores que cero
        mayores_cero.append((DFx[at] > 0).sum())
        menores_cero.append((DFx[at] < 0).sum())

    axs[d-1].text(0.05, 0.75, f"Δα/α > 0:\nΔα/α < 0:", transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.24, 0.75, f"{mayores_cero[0]} \n{menores_cero[0]}", color='tab:blue', transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.3, 0.75, f"{mayores_cero[1]} \n{menores_cero[1]}", color='tab:orange', transform=axs[d-1].transAxes, fontsize=12)


    axs[d-1].text(0.7, 0.58, f"                    \n               ",
         transform=axs[d-1].transAxes,
         fontsize=14,
         bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round',alpha=0.2))

    axs[d-1].text(0.7, 0.65, r"$\Delta\alpha_{med}:$", transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.83, 0.65, f"{DFx.ratea.median():.1f}", color='tab:blue', transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.91, 0.65, f"{DFx.rateb.median():.1f}", color='tab:orange', transform=axs[d-1].transAxes, fontsize=12)

    axs[d-1].text(0.7, 0.59, r"$\sigma_{\Delta\alpha}:$", transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.83, 0.59, f"{DFx.ratea.std():.1f}", color='tab:blue', transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.91, 0.59, f"{DFx.rateb.std():.1f}", color='tab:orange', transform=axs[d-1].transAxes, fontsize=12)

       # axs[d-1].text(0.05, 0.80, f"α < 0: ,{menores_cero[1]}", transform=axs[d-1].transAxes, fontsize=12)

        # Agregar texto con la información
        # Agregar texto con formato LaTeX
#    axs[d-1].text(
#    0.05, 0.85-en/5,
 #   rf"$\alpha > 0: \color{{blue}}{{mayores_cero[0]}},{mayores_cero[1]}$"+f"\n"+
 #   rf"$\alpha < 0: {menores_cero[0]},{menores_cero[1]}$",
 #   transform=axs[d-1].transAxes,
 #   fontsize=12,




    axs[d-1].legend()

axs[0].set_ylabel('Active regions',fontsize=14)
axs[2].set_ylabel('Active regions',fontsize=14)

axs[2].set_xlabel(r'$\Delta \alpha/\alpha$ [1/hour]',fontsize=14)
axs[3].set_xlabel(r'$\Delta \alpha/\alpha$ [1/hour]',fontsize=14)

for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=12)

fig.tight_layout(pad=1.0)

plt.savefig('./plotilt/fracalpha.pdf',dpi=300)

# --- Nueva celda ---

ars=list(set(DF.AR.values))
len(ars)

# --- Nueva celda ---

from diptest import diptest

multiars=[]

#for name in list(set(DF.AR.values)):
for name in ars:


    try:
        DF1=pd.read_csv('./posteriors2/'+str(name)+'_TM3.csv')
    except:
        DF1=pd.read_csv('./posteriors/'+str(name)+'_TM3.csv')

    for i in list(set(DF1['mag'].values)):
        data=DF1[DF1['mag']==i]['alpha'].values
        # Aplicar prueba de dip
        dip_stat, p_value = diptest(data)
      #  print(f"Estadístico de dip: {dip_stat}, p-valor: {p_value}")
        if p_value < 0.05:
            print(str(name)+'----------------------------------')
            multiars.append(name)
            print("La distribución es multimodal (rechazo de unimodalidad).")
            sns.kdeplot(data)
        else:
            pass
#            print("La distribución no es significativamente multimodal.")
    plt.show()

# --- Nueva celda ---

multiars
dic={}
for name in set(multiars):
    dic[name]=multiars.count(name)

# --- Nueva celda ---

dic

# --- Nueva celda ---

from diptest import diptest


DFc=[]
DFu=[]
DFl=[]
DFs=[]

#for name in list(set(DF.AR.values)):
for name in set(multiars):


    try:
        DF1=pd.read_csv('./posteriors2/'+str(name)+'_TM3.csv')
    except:
        DF1=pd.read_csv('./posteriors/'+str(name)+'_TM3.csv')
    #sns.kdeplot(data=DF1,x='alpha',color='black')
    for i in list(set(DF1['mag'].values)):
        data=DF1[DF1['mag']==i]['alpha'].values
        # Aplicar prueba de dip
        dip_stat, p_value = diptest(data)
      #  print(f"Estadístico de dip: {dip_stat}, p-valor: {p_value}")
        if p_value < 0.05:
            mmd=np.median(data)
            DFu.append(DF1[(DF1['mag']==i) & (DF1['alpha'] >mmd)].assign(AR=name))
            DFl.append(DF1[(DF1['mag']==i) & (DF1['alpha'] <mmd)].assign(AR=name))
            if DF1[(DF1['mag']==i) & (DF1['alpha'] <mmd)].std().alpha < DF1[(DF1['mag']==i) & (DF1['alpha'] >mmd)].std().alpha:
                DFs.append(DF1[(DF1['mag']==i) & (DF1['alpha'] <mmd)].assign(AR=name))
            else:
                DFs.append(DF1[(DF1['mag']==i) & (DF1['alpha'] >mmd)].assign(AR=name))


            print(str(name)+'----------------------------------')
            print("La distribución es multimodal (rechazo de unimodalidad).")
        #    sns.kdeplot(data)
        else:
            DFc.append(DF1[DF1['mag']==i].assign(AR=name))
            DFs.append(DF1[DF1['mag']==i].assign(AR=name))
            pass

#            print("La distribución no es significativamente multimodal.")
   # plt.show()

DFc=pd.concat(DFc)
DFu=pd.concat(DFu)
DFl=pd.concat(DFl)
DFs=pd.concat(DFs)


# --- Nueva celda ---

from diptest import diptest
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


DFc=[]
DFu=[]
DFl=[]
DFs=[]

#for name in list(set(DF.AR.values)):
for name in set(multiars):


    try:
        DF1=pd.read_csv('./posteriors2/'+str(name)+'_TM3.csv')
    except:
        DF1=pd.read_csv('./posteriors/'+str(name)+'_TM3.csv')
    #sns.kdeplot(data=DF1,x='alpha',color='black')
    for i in list(set(DF1['mag'].values)):
        data=DF1[DF1['mag']==i]['alpha'].values
        # Aplicar prueba de dip
        dip_stat, p_value = diptest(data)
      #  print(f"Estadístico de dip: {dip_stat}, p-valor: {p_value}")
        if p_value < 0.05:
            # Estimación de densidad usando KDE
            kde = gaussian_kde(data)
            x_vals = np.linspace(min(data) - 1, max(data) + 1, 1000)
            density = kde(x_vals)

            # Detectamos los máximos locales
            peaks, _ = find_peaks(density)

            mmd=np.mean(x_vals[peaks])
            DFu.append(DF1[(DF1['mag']==i) & (DF1['alpha'] >mmd)].assign(AR=name))
            DFl.append(DF1[(DF1['mag']==i) & (DF1['alpha'] <mmd)].assign(AR=name))
            if DF1[(DF1['mag']==i) & (DF1['alpha'] <mmd)].std().alpha < DF1[(DF1['mag']==i) & (DF1['alpha'] >mmd)].std().alpha:
                DFs.append(DF1[(DF1['mag']==i) & (DF1['alpha'] <mmd)].assign(AR=name))
            else:
                DFs.append(DF1[(DF1['mag']==i) & (DF1['alpha'] >mmd)].assign(AR=name))


            print(str(name)+'----------------------------------')
            print("La distribución es multimodal (rechazo de unimodalidad).")
        #    sns.kdeplot(data)
        else:
            DFc.append(DF1[DF1['mag']==i].assign(AR=name))
            DFs.append(DF1[DF1['mag']==i].assign(AR=name))
            pass

#            print("La distribución no es significativamente multimodal.")
   # plt.show()

DFc=pd.concat(DFc)
DFu=pd.concat(DFu)
DFl=pd.concat(DFl)
DFs=pd.concat(DFs)


# --- Nueva celda ---

#for name in list(set(DF.AR.values)):

DFs1=[]
for name in set(multiars):


    try:
        DF1=pd.read_csv('./posteriors2/'+str(name)+'_TM3.csv')
    except:
        DF1=pd.read_csv('./posteriors/'+str(name)+'_TM3.csv')

    for i in list(set(DF1['mag'].values)):

        p1=np.mean(DFc[DFc.AR==name].median().alpha)
        p1=np.nansum([p1,0])
        dis1=np.abs(DFu[(DFu.AR==name)].median().alpha-p1)
        dis2=np.abs(DFl[(DFl.AR==name)].median().alpha-p1)

        if dis1 > dis2:
            DFs1.append(DFl[(DFl.AR==name) & (DFl.mag==i)])
        if dis2 > dis1:
            DFs1.append(DFu[(DFu.AR==name) & (DFu.mag==i)])


DFs1=pd.concat(DFs1)

# --- Nueva celda ---

np.nansum(['nan',0])

# --- Nueva celda ---

DF1[(DF1['mag']==i) & (DF1['alpha'] >mmd)].std().alpha

# --- Nueva celda ---

DF1[(DF1['mag']==i) & (DF1['alpha'] <mmd)].std().alpha

# --- Nueva celda ---



# --- Nueva celda ---

for name in set(multiars):

    ngood=len(DFc[DFc.AR==name].groupby('mag').mean().reset_index())
    nbad=dic[name]
    #ntot=len(DF1.groupby('mag').mean().reset_index())

    if nbad/(nbad+ngood) <0.3:

        sns.lineplot(
            data=DFc[DFc.AR==name],
            x='mag',
            y='alpha',
            color='blue',label='Original',
            estimator='median',
            errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
            err_style='bars',   # o 'bars' para barras verticales
            marker='o'          # opcional para marcar puntos medios
        )
        sns.lineplot(
            data=DFl[DFl.AR==name],
            x='mag',
            y='alpha',
            color='red',label='low med',
            estimator='mean',
            errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
            err_style='bars',   # o 'bars' para barras verticales
            marker='o'          # opcional para marcar puntos medios
        )

        sns.lineplot(
            data=DFu[DFu.AR==name],
            x='mag',
            y='alpha',
            color='green',label='up med',
            estimator='mean',
            errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
            err_style='bars',   # o 'bars' para barras verticales
            marker='o'          # opcional para marcar puntos medios
        )

        sns.lineplot(
            data=DFs1[DFs1.AR==name],
            x='mag',
            y='alpha',
            color='black',label='up med',
            estimator='mean',
            errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
            err_style='bars',   # o 'bars' para barras verticales
            marker='o'          # opcional para marcar puntos medios
        )

        plt.legend()
        plt.title(name)

        plt.show()

# --- Nueva celda ---

for name in set(multiars):
    DFb=pd.concat([DFs1[DFs1.AR==name],DFc[DFc.AR==name]]).sort_values(by='mag', ascending=True)[DFs.keys()[1:-1]].reset_index(drop=True)
    DFb.to_csv(f'./posteriors3/{name}_TM3.csv')

# --- Nueva celda ---



# --- Nueva celda ---

for name in set(multiars):

    try:
        DF1=pd.read_csv('./posteriors2/'+str(name)+'_TM3.csv')
    except:
        DF1=pd.read_csv('./posteriors/'+str(name)+'_TM3.csv')

    ngood=len(DFc[DFc.AR==name].groupby('mag').mean().reset_index())
    nbad=dic[name]
    ntot=len(DF1.groupby('mag').mean().reset_index())

    if nbad/ntot >0.2:





        print(name,len(DFc[DFc.AR==name].groupby('mag').mean().reset_index()),len(DF1.groupby('mag').mean().reset_index()),dic[name])

        sns.scatterplot(data=DF1.groupby('mag').mean().reset_index(),x='mag',y='alpha',color='red')
        sns.scatterplot(data=DF1.groupby('mag').median().reset_index(),x='mag',y='alpha',color='orange')
        sns.scatterplot(data=DFc[DFc.AR==name].groupby('mag').mean().reset_index(),x='mag',y='alpha',color='blue')

      #  plt.title(name)

        plt.show()

# --- Nueva celda ---

DF1

# --- Nueva celda ---

sns.kdeplot(data=DF1,x='alpha')

# --- Nueva celda ---



# --- Nueva celda ---

from scipy.interpolate import interp1d

# --- Nueva celda ---

DF=pd.read_csv('compare-params-TM3-C.csv')
DF0=pd.read_csv('compare-params-TM0.csv')


pixsize=1.98*0.725
DF=DF.assign(cond1=DF.a/(DF.a+DF.R))
DF['gamma'] = 0

# Asignar el valor calculado solo a las filas que cumplen la condición (A > 10)
DF.loc[DF['da']>= DF['cond1'], 'gamma'] = 180*np.arccos((DF.R+DF.a)*(1-DF.da)/DF.R)/np.pi

DF=DF.assign(sepax=lambda x: 2*pixsize*x.R*np.sin(np.pi*x.gamma/180))
DF['sar']=DF['sar'].apply(lambda x: pixsize*x)

DF['alpha']=DF.apply(lambda x: -1*180*np.sign(x.lat)*x.alpha/np.pi,axis=1)
DF['alphab']=DF.apply(lambda x: -1*np.sign(x.lat)*x.alphab,axis=1)
DF[DF.AR == 8913]=DF[DF.AR == 8913].assign(fint=DF[DF.AR == 8913].flux/np.max(DF[DF.AR == 8913].flux))
DF=DF.assign(rot= lambda x: np.gradient(x.alpha,1.5*x.mag))
DF=DF.assign(rotb= lambda x: np.gradient(x.alphab,1.5*x.mag))

DF0['alpha']=DF0.apply(lambda x: -1*180*np.sign(x.lat)*x.alpha/np.pi,axis=1)
DF0['alphab']=DF0.apply(lambda x: -1*np.sign(x.lat)*x.alphab,axis=1)
DF0[DF0.AR == 8913]=DF0[DF0.AR == 8913].assign(fint=DF0[DF0.AR == 8913].flux/np.max(DF0[DF0.AR == 8913].flux))
DF0[DF0.AR == 8214]=DF0[DF0.AR == 8214].assign(fint=DF0[DF0.AR == 8214].flux/np.max(DF0[DF0.AR == 8214].flux))
DF0=DF0.assign(rot= lambda x: np.gradient(x.alpha,x.mag)/1.5)
DF0=DF0.assign(rotb= lambda x: np.gradient(x.alphab,x.mag)/1.5)

# --- Nueva celda ---

DF[DF.AR==10268]

# --- Nueva celda ---

ars=list(set(DF.AR.values))
ars0=list(set(DF0.AR.values))

# --- Nueva celda ---

#Select ARs in which rotation is larger that 4 degree per hour (probably problematic cases)
DFmax=np.abs(DF).groupby('AR').max().reset_index()
listar=list(DFmax[np.abs(DFmax.rot) >4].AR)
#listar=list(DFmax[(DFmax.difalpha >40)*(np.abs(DFmax.rot) >2)].AR)

# --- Nueva celda ---

len(listar)

# --- Nueva celda ---

DFxx=[]

#Analize ARs with large rotation, an cut first part of the emergence to reduce their number.

for name in listar:
    DF2 = DF[DF.AR == name]

    #if DF2.fint.min() >= lower_lim:
    #    print(f"Región {name} no llega a f_norm = {lower_lim}, se excluye.")
    #    continue

    limax=DF2[np.abs(DF2.rot)>4].max().mag
    DFmax=np.abs(DF2).groupby('AR').max().reset_index()

    if (np.abs(DFmax.rot.values) >4) & (DF2[DF2.mag==limax].fint.values <=0.3):
        print(name)
        sns.lineplot(data=DF2,x='fint',y='alpha')
        sns.lineplot(data=DF2[DF2.mag>limax],x='fint',y='alpha')

       # sns.lineplot(data=DF2,x='fint',y='alphab')
        plt.show()

        DFxx.append(DF2[DF2.mag>limax])

DFxx=pd.concat(DFxx)


# --- Nueva celda ---

ars2=list(set(listar)^set(ars))

# --- Nueva celda ---

len(ars2)

# --- Nueva celda ---

DFb=pd.concat([DFxx,DF[DF['AR'].isin(ars2)]])
len(list(set(DFb.AR.values)))

# --- Nueva celda ---

len(ars2)

# --- Nueva celda ---

tilta = []
tiltb = []

lower_lim = 0.5
DFx = []

time_grid = np.linspace(0, 1.0, 11)  # Grilla común

#for name in ars2:
for name in list(set(DFb.AR.values)):

    DF2 = DFb[DFb.AR == name]

    #if DF2.fint.min() >= lower_lim:
    #    print(f"Región {name} no llega a f_norm = {lower_lim}, se excluye.")
    #    continue

    lower_lim =   DF2.fint.min()+0.2
    # Normalización basada en el valor de mag en el punto de máximo flujo normalizado
    idx_max = DF2.fint.idxmax()
    magmax = DF2.loc[idx_max].mag

    idmax=DF2.sar.idxmax()
    sarmax=DF2.loc[idmax].mag

    # (opcional) linealizar si se desea ajustar f_norm como función de mag/magmax
    DF2p = DF2[DF2.fint < lower_lim]
    x_pre = DF2p.mag.values / magmax
    y_pre = DF2p.fint.values

    slope, intercept, _, _, _ = stats.linregress(y_pre, x_pre)
    # Definir eje de tiempo normalizado
    t_norm = (DF2.mag.values/ magmax - intercept) / (1 - intercept)

    if intercept >0:
      print(f'AR {name} t_norm 0 is lower than 0 ***{intercept}***')
      plt.plot(y_pre,x_pre)
      plt.plot(np.arange(0,0.2,0.1),slope*np.arange(0,0.2,0.1)+intercept,color='r')
      plt.show()

    DF2s=DF2[DF2.sar/DF2.loc[idmax].sar<0.8]
    x_pre = DF2s.mag.values / sarmax
    y_pre = DF2s.sar.values/DF2.loc[idmax].sar
    slope, intercept, _, _, _ = stats.linregress(y_pre, x_pre)
    # Definir eje de tiempo normalizado
    d_norm = (DF2.mag.values - intercept) / (sarmax - intercept)







 #   DFx.append(pd.DataFrame({'AR':name,'fn':int_fn(time_grid),
 #                            'alpha':int_alpha(time_grid),'alphab':int_alphab(time_grid),'t_grid':time_grid,'t_norm':np.linspace(0,1,11),'lat':DF2.lat.mean()}))
    tilta.append(DF2.alpha.values)
    tiltb.append(DF2.alphab.values)

    DFx.append(pd.DataFrame({'AR':name,'alpha':DF2.alpha.values,'alphab':DF2.alphab.values,'t_norm':t_norm,
                             'lat':DF2.lat.mean(),'fn':DF2.fint.values,'Nt':DF2.N0.values,
                             'sar':DF2.sar.values,'sepax':DF2.sepax.values,
                             'mag':DF2.mag.values,'flux':DF2.flux.values,'fint':DF2.fint.values,'d_norm':d_norm}))

DFx=pd.concat(DFx)

# --- Nueva celda ---

DFx[DFx.t_norm <0]

# --- Nueva celda ---

len(tilta)

# --- Nueva celda ---

tilta = []
tiltb = []

lower_lim = 0.3
DFx0 = []

time_grid = np.linspace(0, 1.0, 11)  # Grilla común

for name in ars0:
    DF0[DF0.AR == name]=DF0[DF0.AR == name].assign(fint=DF0[DF0.AR == name].flux/np.max(DF0[DF0.AR == name].flux))

    DF2 = DF0[DF0.AR == name]

    if DF2.fint.min() >= lower_lim:
        print(f"Región {name} no llega a f_norm = {lower_lim}, se excluye.")
        continue

    # Normalización basada en el valor de mag en el punto de máximo flujo normalizado
    idx_max = DF2.fint.idxmax()
    magmax = DF2.loc[idx_max].mag

    # (opcional) linealizar si se desea ajustar f_norm como función de mag/magmax
    DF2p = DF2[DF2.fint < lower_lim]
    x_pre = DF2p.mag.values / magmax
    y_pre = DF2p.fint.values

    slope, intercept, _, _, _ = stats.linregress(y_pre, x_pre)

    # Definir eje de tiempo normalizado
    t_norm = (DF2.mag.values - intercept) / (magmax - intercept)

    # Interpolación del tilt (alpha) en función de t_norm
    try:
        int_alpha = interp1d(t_norm, DF2.alpha.values, bounds_error=False, fill_value=np.nan)
        int_alphab = interp1d(t_norm, DF2.alphab.values, bounds_error=False, fill_value=np.nan)
        int_fn=interp1d(t_norm, DF2.fint, bounds_error=False, fill_value=np.nan)

     #   tilta.append(int_alpha(time_grid))
     #   tiltb.append(int_alphab(time_grid))
    except Exception as e:
        print(f"Interpolación fallida en región {name}: {e}")

 #   DFx.append(pd.DataFrame({'AR':name,'fn':int_fn(time_grid),
 #                            'alpha':int_alpha(time_grid),'alphab':int_alphab(time_grid),'t_grid':time_grid,'t_norm':np.linspace(0,1,11),'lat':DF2.lat.mean()}))
    tilta.append(DF2.alpha.values)
    tiltb.append(DF2.alphab.values)

    DFx0.append(pd.DataFrame({'AR':name,'alpha':DF2.alpha.values,'alphab':DF2.alphab.values,'t_norm':t_norm,
                             'lat':DF2.lat.mean(),'fn':DF2.fint.values}))

DFx0=pd.concat(DFx0)

# --- Nueva celda ---

DFx0b=pd.DataFrame({'t_norm':DFx0.t_norm.values,'lat':DFx0.lat.values,'AR':DFx0.AR.values,
                    'fn':DFx0.fn.values,'variable':'alpha0','value':DFx0.alpha.values})[::3]

# --- Nueva celda ---

DFx

# --- Nueva celda ---

DFx2=DFx.assign(frac=lambda x: np.sign(x.alpha*x.alphab)*np.abs(x.alpha-x.alphab)/np.abs(x.alphab))
DFx2=DFx2.assign(t_mean=lambda x: round(10*x.t_norm)/10)

# --- Nueva celda ---

DFx2=DFx.assign(frac=lambda x: np.sign(x.alpha*x.alphab)*np.abs(x.alpha-x.alphab)/np.abs(x.alphab))
DFx2=DFx2.assign(t_mean=lambda x: round(10*x.t_norm)/10)

DFx2=DFx2[DFx2.t_mean<=1]


DFmax_D = (
    DFx2[DFx2.t_mean == 1]
    .groupby("AR")[["sar", "sepax"]]
    .mean()
    .reset_index()
    .rename(columns={"sar": "sar_max", "sepax": "sepax_max"})
)

DFx2 = DFx2.merge(DFmax_D, on="AR", how="left")


sns.scatterplot(data=DFx2.assign(sepax=lambda x: x.sepax/x.sepax_max),x='t_norm',y='sepax',alpha=0.2)
sns.scatterplot(data=DFx2.assign(sar=lambda x: x.sar/x.sar_max),x='t_norm',y='sar',alpha=0.2)

DFx2=DFx2.groupby(['AR','t_mean']).median().reset_index()


sns.lineplot(data=DFx2.assign(sepax=lambda x: x.sepax/x.sepax_max),x='t_mean',y='sepax',estimator='median',
errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='band',   # o 'bars' para barras verticales
marker='o',label='Axial separation'
          )
sns.lineplot(data=DFx2.assign(sar=lambda x: x.sar/x.sar_max),x='t_mean',y='sar',estimator='median',
errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='band',   # o 'bars' para barras verticales
marker='o',label='Barycenters'

             )

plt.legend()
plt.ylim(0,1.2)
plt.ylabel(r'D/D$_\mathrm{t=1}$')
plt.xlabel(r'$t_{norm}$')
plt.show()


# --- Nueva celda ---



# --- Nueva celda ---



# --- Nueva celda ---

DFx

# --- Nueva celda ---

DFx2=DFx.assign(frac=lambda x: x.alpha-x.alphab)
DFx2=DFx2.assign(t_mean=lambda x: round(10*x.t_norm)/10)


g=sns.scatterplot(data=DFx2,x='t_norm',y='fint',alpha=0.2)

DFx2=DFx2.groupby(['AR','t_mean']).mean().reset_index()
g=sns.lineplot(
data=DFx2,
x='t_mean',
y='fint',
estimator='mean',
errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='band',   # o 'bars' para barras verticales
marker='o'          # opcional para marcar puntos medios
)
#   sns.lineplot(data=DFx2.groupby(['t_norm','variable']).median().reset_index(),x='t_norm',y='value',hue='variable')
plt.xlim(0,1)

# --- Nueva celda ---


DFx2=DFx.assign(frac=lambda x: x.alpha-x.alphab)
DFx2=DFx2.assign(t_mean=lambda x: round(10*x.t_norm)/10)


g=sns.scatterplot(data=DFx2,x='t_norm',y='frac',alpha=0.2)

DFx2=DFx2.groupby(['AR','t_mean']).mean().reset_index()
g=sns.lineplot(
data=DFx2,
x='t_mean',
y='frac',
estimator='mean',
errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='band',   # o 'bars' para barras verticales
marker='o'          # opcional para marcar puntos medios
)
#   sns.lineplot(data=DFx2.groupby(['t_norm','variable']).median().reset_index(),x='t_norm',y='value',hue='variable')
plt.xlim(0,1)
plt.ylim(-40,40)

plt.ylabel(r'$\alpha_{mod}-\alpha_{bar}$  [deg]')
plt.xlabel(r'$t_{norm}$')
plt.show()


sns.histplot(data=DFx2[DFx2.t_mean.isin([0.1,0.5,0.9])],x='frac',hue='t_mean',palette='deep',
             binrange=(-50,50)
             ,kde=True,multiple='layer')

plt.xlim(-50,50)
plt.xlabel(r'$\alpha_{mod}-\alpha_{bar}$  [deg]')

plt.show()


# --- Nueva celda ---

DFx['stp'] = DFx.groupby('AR')['mag'].transform(lambda x: np.gradient(x))

# --- Nueva celda ---

DFx[DFx.AR==10268]

# --- Nueva celda ---

alpha_max_fn=DFx.groupby('AR').apply(lambda x: x.nlargest(1, 'fn'))

# Paso 2: crear la nueva columna normalizada
DFx=DFx.assign(alphan=lambda row: row['alpha'] / alpha_max_fn.loc[row['AR']].alpha.values)
DFx=DFx.assign(alphabn=lambda row: row['alphab'] / alpha_max_fn.loc[row['AR']].alphab.values)

# --- Nueva celda ---

sns.histplot(np.abs(DFx.groupby('AR').mean().reset_index()),x='lat',binrange=(0,40),bins=20,kde=True)

# --- Nueva celda ---

estim='median'


f=plt.figure(figsize=(6,3))

DFx2=DFx.assign(frac=lambda x: np.abs(x.alpha-x.alphab))
DFx2=DFx2.assign(t_mean=lambda x: round(10*x.t_norm)/10)
DFx2=DFx2[DFx2.t_mean <=1]


DFx2=DFx2[(np.abs(DFx2.lat) >= 16) & (np.abs(DFx2.lat) <= 17)]

arsinbin=DFx2.AR.unique()
print(len(arsinbin))
g=sns.lineplot(
data=pd.melt(DFx2,id_vars=['AR','t_mean'],value_vars=['alpha','alphab'],var_name='variable'),
x='t_mean',
y='value',
                hue='variable',
estimator=estim,
errorbar='ci',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='bars',   # o 'bars' para barras verticales
marker='o',          # opcional para marcar puntos medios
linewidth=3,markersize=10,alpha=0.5)

plt.title(f'latrange: [16-17]/{len(arsinbin):.0f} ARs')

plt.show()

f=plt.figure(figsize=(6,3))


f_tresh=np.median(DFx2[DFx2.t_mean == 1]['flux'])
DFmax=DFx2[DFx2.t_mean==1].groupby('AR').mean().reset_index()
ARup=DFmax[DFmax.flux > f_tresh].AR.unique()
ARlow=DFmax[DFmax.flux <= f_tresh].AR.unique()


g=sns.lineplot(
data=pd.melt(DFx2[DFx2.AR.isin(ARup)],id_vars=['AR','t_mean'],value_vars=['alpha','alphab'],var_name='variable'),
x='t_mean',
y='value',
                hue='variable',
estimator=estim,
errorbar='ci',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='bars',   # o 'bars' para barras verticales
marker='o',          # opcional para marcar puntos medios
linewidth=3,markersize=10,alpha=0.5)

plt.title(f'latrange: [16-17] / largeflux / {len(ARup)} AR')

plt.show()

f=plt.figure(figsize=(6,3))

g=sns.lineplot(
data=pd.melt(DFx2[DFx2.AR.isin(ARlow)],id_vars=['AR','t_mean'],value_vars=['alpha','alphab'],var_name='variable'),
x='t_mean',
y='value',
                hue='variable',
estimator=estim,
errorbar='ci',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='bars',   # o 'bars' para barras verticales
marker='o',          # opcional para marcar puntos medios
linewidth=3,markersize=10,alpha=0.5)

plt.title(f'latrange: [16-17] / lowflux / {len(ARlow)} AR')

plt.show()

f=plt.figure(figsize=(6,3))

g=sns.lineplot(
data=pd.melt(DFx2[np.abs(DFx2.Nt) > np.median(np.abs(DFx2.Nt))],id_vars=['AR','t_mean'],value_vars=['alpha','alphab'],var_name='variable'),
x='t_mean',
y='value',
                hue='variable',
estimator=estim,
errorbar='ci',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='bars',   # o 'bars' para barras verticales
marker='o',          # opcional para marcar puntos medios
linewidth=3,markersize=10,alpha=0.5)

plt.title(f'latrange: [16-17] / largeNt / {len(DFx2[np.abs(DFx2.Nt) > np.median(np.abs(DFx2.Nt))].AR.unique())} AR')


plt.show()

f=plt.figure(figsize=(6,3))

g=sns.lineplot(
data=pd.melt(DFx2[np.abs(DFx2.Nt) <= np.median(np.abs(DFx2.Nt))],id_vars=['AR','t_mean'],value_vars=['alpha','alphab'],var_name='variable'),
x='t_mean',
y='value',
                hue='variable',
estimator=estim,
errorbar='ci',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='bars',   # o 'bars' para barras verticales
marker='o',          # opcional para marcar puntos medios
linewidth=3,markersize=10,alpha=0.5)

plt.title(f'latrange: [16-17] / lowNt / {len(DFx2[np.abs(DFx2.Nt) <= np.median(np.abs(DFx2.Nt))].AR.unique())} AR')

plt.show()


DFx2=DFx2.assign(tw=lambda x: np.abs(x.Nt)*x.flux**2)
DFmax=DFx2[DFx2.t_mean==1].groupby('AR').mean().reset_index()
ARup=DFmax[DFmax.tw >= np.median(DFmax.tw)].AR.unique()
ARlow=DFmax[DFmax.tw < np.median(DFmax.tw)].AR.unique()

f=plt.figure(figsize=(6,3))

g=sns.lineplot(
data=pd.melt(DFx2[DFx2.AR.isin(ARup)],id_vars=['AR','t_mean'],value_vars=['alpha','alphab'],var_name='variable'),
x='t_mean',
y='value',
                hue='variable',
estimator=estim,
errorbar='ci',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='bars',   # o 'bars' para barras verticales
marker='o',          # opcional para marcar puntos medios
linewidth=3,markersize=10,alpha=0.5)

plt.title(f'latrange: [16-17] / largetwist / {len(ARup)} AR')

plt.show()

f=plt.figure(figsize=(6,3))

g=sns.lineplot(
data=pd.melt(DFx2[DFx2.AR.isin(ARlow)],id_vars=['AR','t_mean'],value_vars=['alpha','alphab'],var_name='variable'),
x='t_mean',
y='value',
                hue='variable',
estimator=estim,
errorbar='ci',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='bars',   # o 'bars' para barras verticales
marker='o',          # opcional para marcar puntos medios
linewidth=3,markersize=10,alpha=0.5)

plt.title(f'latrange: [16-17] / lowtwist / {len(ARlow)} AR')

plt.show()



# --- Nueva celda ---

estim='median'


f=plt.subplot(figsize=(6,9))

DFx2=DFx.assign(frac=lambda x: np.abs(x.alpha-x.alphab))
DFx2=DFx2.assign(t_mean=lambda x: round(10*x.t_norm)/10)
DFx2=DFx2[DFx2.t_mean <=1]

DFx2=DFx2.assign(tw=lambda x: np.abs(x.Nt)*x.flux**2)
DFmax=DFx2[DFx2.t_mean==1].groupby('AR').mean().reset_index()
ARup=DFmax[DFmax.tw >= np.median(DFmax.tw)].AR.unique()
ARlow=DFmax[DFmax.tw < np.median(DFmax.tw)].AR.unique()


DFx2=DFx2[(np.abs(DFx2.lat) >= 16) & (np.abs(DFx2.lat) <= 17)]

arsinbin=DFx2.AR.unique()
print(len(arsinbin))
g=sns.lineplot(
data=pd.melt(DFx2,id_vars=['AR','t_mean'],value_vars=['alpha','alphab'],var_name='variable'),
x='t_mean',
y='value',
                hue='variable',
estimator=estim,
errorbar='ci',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='bars',   # o 'bars' para barras verticales
marker='o',          # opcional para marcar puntos medios
linewidth=3,markersize=10,alpha=0.5)

plt.title(f'latrange: [16-17]/{len(arsinbin):.0f} ARs')

plt.show()


f=plt.figure(figsize=(6,3))

g=sns.lineplot(
data=pd.melt(DFx2[DFx2.AR.isin(ARup)],id_vars=['AR','t_mean'],value_vars=['alpha','alphab'],var_name='variable'),
x='t_mean',
y='value',
                hue='variable',
estimator=estim,
errorbar='ci',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='bars',   # o 'bars' para barras verticales
marker='o',          # opcional para marcar puntos medios
linewidth=3,markersize=10,alpha=0.5)

plt.title(f'latrange: [16-17] / largetwist / {len(DFx2[DFx2.AR.isin(ARup)].AR.unique())} AR')

plt.show()

f=plt.figure(figsize=(6,3))

g=sns.lineplot(
data=pd.melt(DFx2[DFx2.AR.isin(ARlow)],id_vars=['AR','t_mean'],value_vars=['alpha','alphab'],var_name='variable'),
x='t_mean',
y='value',
                hue='variable',
estimator=estim,
errorbar='ci',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='bars',   # o 'bars' para barras verticales
marker='o',          # opcional para marcar puntos medios
linewidth=3,markersize=10,alpha=0.5)

plt.title(f'latrange: [16-17] / lowtwist / {len(DFx2[DFx2.AR.isin(ARlow)].AR.unique())} AR')

plt.show()



# --- Nueva celda ---

DFx2=DFx.assign(frac=lambda x: np.abs(x.alpha-x.alphab))
DFx2=DFx2.assign(t_mean=lambda x: round(10*x.t_norm)/10)

len(DFx2.AR.unique())

# --- Nueva celda ---

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

estim = 'median'

lat_ranges = [(0,9),(10, 14), (15, 19), (20, 25),(26,39)]   # 👈 tus tres rangos de latitud

fig, axes = plt.subplots(5, 3, figsize=(12, 12), sharex=True, sharey=True)

for i, (latmin, latmax) in enumerate(lat_ranges):

    # Filtrar rango de latitudes
    DFx2 = DFx.assign(frac=lambda x: np.abs(x.alpha - x.alphab))
    DFx2 = DFx2.assign(t_mean=lambda x: round(10 * x.t_norm) / 10)
    DFx2 = DFx2[DFx2.t_mean <= 1]
    DFx2 = DFx2.assign(tw=lambda x: np.abs(x.Nt) * x.flux**2)
        # ARs de alto y bajo twist
    DFmax = DFx2[DFx2.t_mean == 1].groupby('AR').mean().reset_index()
    ARup = DFmax[DFmax.tw >= np.median(DFmax.tw)].AR.unique()
    ARlow = DFmax[DFmax.tw < np.median(DFmax.tw)].AR.unique()


    DFx2 = DFx2[(np.abs(DFx2.lat) >= latmin) & (np.abs(DFx2.lat) <= latmax)]



    # Panel 1: todas las ARs
    g = sns.lineplot(
        data=pd.melt(DFx2, id_vars=['AR','t_mean'],
                     value_vars=['alpha','alphab'], var_name='variable'),
        x='t_mean', y='value', hue='variable',
        estimator=estim, errorbar='ci', err_style='bars',
        marker='o', linewidth=2, markersize=6, alpha=0.6,
        ax=axes[i,0]
    )
    axes[i,0].set_title(f'lat: [{latmin}-{latmax}] / {DFx2.AR.nunique()} ARs')

    # Panel 2: alto twist
    g = sns.lineplot(
        data=pd.melt(DFx2[DFx2.AR.isin(ARup)], id_vars=['AR','t_mean'],
                     value_vars=['alpha','alphab'], var_name='variable'),
        x='t_mean', y='value', hue='variable',
        estimator=estim, errorbar='ci', err_style='bars',
        marker='o', linewidth=2, markersize=6, alpha=0.6,
        ax=axes[i,1]
    )
    axes[i,1].set_title(f'lat: [{latmin}-{latmax}] / high twist / {len(DFx2[DFx2.AR.isin(ARup)].AR.unique())} ARs')

    # Panel 3: bajo twist
    g = sns.lineplot(
        data=pd.melt(DFx2[DFx2.AR.isin(ARlow)], id_vars=['AR','t_mean'],
                     value_vars=['alpha','alphab'], var_name='variable'),
        x='t_mean', y='value', hue='variable',
        estimator=estim, errorbar='ci', err_style='bars',
        marker='o', linewidth=2, markersize=6, alpha=0.6,
        ax=axes[i,2]
    )
    axes[i,2].set_title(f'lat: [{latmin}-{latmax}] / low twist / {len(DFx2[DFx2.AR.isin(ARlow)].AR.unique())} ARs')

# Etiquetas globales
for ax in axes.flat:
    ax.axhline(0, linestyle='--', color='gray', alpha=0.6)
    ax.set_xlim(0,1)
    ax.set_ylim(-20,40)
    ax.set_xlabel(r'$t_{norm}$')
    ax.set_ylabel(r'$\alpha$ [deg]')

plt.tight_layout()
plt.show()


# --- Nueva celda ---

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

estim = 'median'

lat_ranges = [(0,9),(10, 14), (15, 19), (20, 25),(26,39)]   # 👈 tus tres rangos de latitud

fig, axes = plt.subplots(5, 3, figsize=(12, 12), sharex=True, sharey=True)

for i, (latmin, latmax) in enumerate(lat_ranges):

    # Filtrar rango de latitudes
    DFx2 = DFx.assign(frac=lambda x: np.abs(x.alpha - x.alphab))
    DFx2 = DFx2.assign(t_mean=lambda x: round(10 * x.t_norm) / 10)
    DFx2 = DFx2[DFx2.t_mean <= 1]
    DFx2 = DFx2.assign(tw=lambda x: np.abs(x.Nt) * x.flux**2)
        # ARs de alto y bajo twist
    DFmax = DFx2[DFx2.t_mean == 1].groupby('AR').mean().reset_index()
    ARup = DFmax[DFmax.flux >= np.median(DFmax.flux)].AR.unique()
    ARlow = DFmax[DFmax.flux < np.median(DFmax.flux)].AR.unique()


    DFx2 = DFx2[(np.abs(DFx2.lat) >= latmin) & (np.abs(DFx2.lat) <= latmax)]



    # Panel 1: todas las ARs
    g = sns.lineplot(
        data=pd.melt(DFx2, id_vars=['AR','t_mean'],
                     value_vars=['alpha','alphab'], var_name='variable'),
        x='t_mean', y='value', hue='variable',
        estimator=estim, errorbar='ci', err_style='bars',
        marker='o', linewidth=2, markersize=6, alpha=0.6,
        ax=axes[i,0]
    )
    axes[i,0].set_title(f'lat: [{latmin}-{latmax}] / {DFx2.AR.nunique()} ARs')

    # Panel 2: alto twist
    g = sns.lineplot(
        data=pd.melt(DFx2[DFx2.AR.isin(ARup)], id_vars=['AR','t_mean'],
                     value_vars=['alpha','alphab'], var_name='variable'),
        x='t_mean', y='value', hue='variable',
        estimator=estim, errorbar='ci', err_style='bars',
        marker='o', linewidth=2, markersize=6, alpha=0.6,
        ax=axes[i,1]
    )
    axes[i,1].set_title(f'lat: [{latmin}-{latmax}] / high flux / {len(DFx2[DFx2.AR.isin(ARup)].AR.unique())} ARs')

    # Panel 3: bajo twist
    g = sns.lineplot(
        data=pd.melt(DFx2[DFx2.AR.isin(ARlow)], id_vars=['AR','t_mean'],
                     value_vars=['alpha','alphab'], var_name='variable'),
        x='t_mean', y='value', hue='variable',
        estimator=estim, errorbar='ci', err_style='bars',
        marker='o', linewidth=2, markersize=6, alpha=0.6,
        ax=axes[i,2]
    )
    axes[i,2].set_title(f'lat: [{latmin}-{latmax}] / low flux / {len(DFx2[DFx2.AR.isin(ARlow)].AR.unique())} ARs')

# Etiquetas globales
for ax in axes.flat:
    ax.axhline(0, linestyle='--', color='gray', alpha=0.6)
    ax.set_xlim(0,1)
    ax.set_ylim(-20,40)
    ax.set_xlabel(r'$t_{norm}$')
    ax.set_ylabel(r'$\alpha$ [deg]')

plt.tight_layout()
plt.show()


# --- Nueva celda ---

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

estim = 'median'

lat_ranges = [(0,9),(10, 14), (15, 19), (20, 25),(26,39)]   # 👈 tus tres rangos de latitud

fig, axes = plt.subplots(5, 3, figsize=(12, 12), sharex=True, sharey=True)

DFx=DFx.assign(rot= lambda x: np.gradient(x.alpha,x.mag)/1.5)
DFx=DFx.assign(rotb= lambda x: np.gradient(x.alphab,x.mag)/1.5)

for i, (latmin, latmax) in enumerate(lat_ranges):

    # Filtrar rango de latitudes
    DFx2 = DFx.assign(frac=lambda x: np.abs(x.alpha - x.alphab))
    DFx2 = DFx2.assign(t_mean=lambda x: round(10 * x.t_norm) / 10)
    DFx2 = DFx2[DFx2.t_mean <= 1]
    DFx2 = DFx2.assign(tw=lambda x: np.abs(x.Nt) * x.flux**2)
        # ARs de alto y bajo twist
    DFmax = DFx2[DFx2.t_mean == 1].groupby('AR').mean().reset_index()
    ARup = DFmax[DFmax.flux >= np.median(DFmax.flux)].AR.unique()
    ARlow = DFmax[DFmax.flux < np.median(DFmax.flux)].AR.unique()


    DFx2 = DFx2[(np.abs(DFx2.lat) >= latmin) & (np.abs(DFx2.lat) <= latmax)]



    # Panel 1: todas las ARs
    g = sns.lineplot(
        data=pd.melt(DFx2, id_vars=['AR','t_mean'],
                     value_vars=['rot','rotb'], var_name='variable'),
        x='t_mean', y='value', hue='variable',
        estimator=estim, errorbar='ci', err_style='bars',
        marker='o', linewidth=2, markersize=6, alpha=0.6,
        ax=axes[i,0]
    )
    axes[i,0].set_title(f'lat: [{latmin}-{latmax}] / {DFx2.AR.nunique()} ARs')

    # Panel 2: alto twist
    g = sns.lineplot(
        data=pd.melt(DFx2[DFx2.AR.isin(ARup)], id_vars=['AR','t_mean'],
                     value_vars=['rot','rotb'], var_name='variable'),
        x='t_mean', y='value', hue='variable',
        estimator=estim, errorbar='ci', err_style='bars',
        marker='o', linewidth=2, markersize=6, alpha=0.6,
        ax=axes[i,1]
    )
    axes[i,1].set_title(f'lat: [{latmin}-{latmax}] / high flux / {len(DFx2[DFx2.AR.isin(ARup)].AR.unique())} ARs')

    # Panel 3: bajo twist
    g = sns.lineplot(
        data=pd.melt(DFx2[DFx2.AR.isin(ARlow)], id_vars=['AR','t_mean'],
                     value_vars=['rot','rotb'], var_name='variable'),
        x='t_mean', y='value', hue='variable',
        estimator=estim, errorbar='ci', err_style='bars',
        marker='o', linewidth=2, markersize=6, alpha=0.6,
        ax=axes[i,2]
    )
    axes[i,2].set_title(f'lat: [{latmin}-{latmax}] / low flux / {len(DFx2[DFx2.AR.isin(ARlow)].AR.unique())} ARs')

# Etiquetas globales
for ax in axes.flat:
    ax.axhline(0, linestyle='--', color='gray', alpha=0.6)
    ax.set_xlim(0,1)
    ax.set_ylim(-1,1)
    ax.set_xlabel(r'$t_{norm}$')
    ax.set_ylabel(r'$\alpha$ [deg]')

plt.tight_layout()
plt.show()

# --- Nueva celda ---

sns.histplot(DFx,)

# --- Nueva celda ---

DFx2 = DFx.assign(frac=lambda x: np.abs(x.alpha - x.alphab))
DFx2 = DFx2.assign(t_mean=lambda x: round(10 * x.t_norm) / 10)
DFx2 = DFx2[DFx2.t_mean <= 1]
DFx2 = DFx2.assign(tw=lambda x: np.abs(x.Nt) * x.flux**2)
    # ARs de alto y bajo twist

DFmax = DFx2[DFx2.t_mean == 1].groupby('AR').mean().reset_index()

sns.histplot(data=DFmax,x='tw',bins=10)
plt.axvline(np.median(DFmax.tw),color='red')

plt.xlabel('Twist [Mx^2]')
plt.ylabel('ARs')

plt.show()

sns.histplot(data=DFmax,x='flux',bins=10)
plt.axvline(np.median(DFmax.flux),color='red')

plt.xlabel('Max Flux [Mx]')
plt.ylabel('ARs')

plt.show()

# --- Nueva celda ---

DFmax

# --- Nueva celda ---

DFx2

# --- Nueva celda ---

DFx['stp'] = DFx.groupby('AR')['mag'].transform(lambda x: np.gradient(x))

DFx=DFx.assign(rot= lambda x: np.gradient(x.alpha,1.5*x.mag))
DFx=DFx.assign(rotb= lambda x: np.gradient(x.alphab,1.5*x.mag))


DFx["rot_cum"]  = DFx.groupby("AR")["rot"].cumsum()
DFx["rotb_cum"] = DFx.groupby("AR")["rotb"].cumsum()

DFrot=DFx.groupby('AR').apply(lambda x: x.nlargest(1, 'fn')).reset_index(drop=True)
DFrot=DFrot.assign(rot_cum=lambda x: x.rot_cum*1.5*x.stp)
DFrot=DFrot.assign(rotb_cum=lambda x: x.rotb_cum*1.5*x.stp)


# --- Nueva celda ---

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

estim = 'median'

lat_ranges = [(0,39)]   # 👈 tus tres rangos de latitud

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

DFx=DFx.assign(rot= lambda x: np.gradient(x.alpha,x.mag)/1.5)
DFx=DFx.assign(rotb= lambda x: np.gradient(x.alphab,x.mag)/1.5)


# Filtrar rango de latitudes
DFx2 = DFx.assign(frac=lambda x: np.abs(x.alpha - x.alphab))
DFx2 = DFx2.assign(t_mean=lambda x: round(10 * x.t_norm) / 10)
DFx2 = DFx2[DFx2.t_mean <= 1]
DFx2 = DFx2.assign(tw=lambda x: np.abs(x.Nt) * x.flux**2)
    # ARs de alto y bajo twist
DFmax = DFx2[DFx2.t_mean == 1].groupby('AR').mean().reset_index()
ARup = DFmax[DFmax.flux >= np.median(DFmax.flux)].AR.unique()
ARlow = DFmax[DFmax.flux < np.median(DFmax.flux)].AR.unique()

ARpos1=DFmax[DFmax.rot_cum >0].AR.unique()
ARneg1=DFmax[DFmax.rot_cum <0].AR.unique()

ARpos2=DFmax[DFmax.rotb_cum >0].AR.unique()
ARneg2=DFmax[DFmax.rotb_cum <0].AR.unique()


for i, ars in enumerate([DFx.AR.unique(),ARup,ARlow]):



  df1=DFx2[DFx2.AR.isin(ars)]

  g = sns.lineplot(
    data=df1[df1.AR.isin(ARpos1)],
    x='t_mean', y='rot',color='tab:blue',
    estimator=estim, errorbar='ci', err_style='bars',
    marker='o', linewidth=2, markersize=6, alpha=0.6,
    ax=axes[i],label=r'$\gamma_\mathrm{mod}>0$'
)
  g = sns.lineplot(
  data=df1[df1.AR.isin(ARneg1)],
  x='t_mean', y='rot',color='tab:blue',
  estimator=estim, errorbar='ci', err_style='bars',
  marker='^', linewidth=2,linestyle='dashed', markersize=6, alpha=0.6,
  ax=axes[i],label=r'$\gamma_\mathrm{mod}$<0'
)

  g = sns.lineplot(
    data=df1[df1.AR.isin(ARpos2)],
    x='t_mean', y='rotb',color='tab:orange',
    estimator=estim, errorbar='ci', err_style='bars',
    marker='o', linewidth=2, markersize=6, alpha=0.6,
    ax=axes[i],label=r'$\gamma_\mathrm{bar}>0$'
)
  g = sns.lineplot(
  data=df1[df1.AR.isin(ARneg2)],
  x='t_mean', y='rotb',color='tab:orange',
  estimator=estim, errorbar='ci', err_style='bars',
  marker='^', linewidth=2,linestyle='dashed', markersize=6, alpha=0.6,
  ax=axes[i],label=r'$\gamma_\mathrm{bar}<0$'
)


  axes[i].set_title(f"{['All','High flux','Low flux'][i]} ARs")

  axes[i].text(0.7,0.5,f'{len(df1[df1.AR.isin(ARpos1)].AR.unique())}',color='tab:blue')
  axes[i].text(0.8,0.5,f'{len(df1[df1.AR.isin(ARpos2)].AR.unique())}',color='tab:orange')

  axes[i].text(0.7,-0.5,f'{len(df1[df1.AR.isin(ARneg1)].AR.unique())}',color='tab:blue')
  axes[i].text(0.8,-0.5,f'{len(df1[df1.AR.isin(ARneg2)].AR.unique())}',color='tab:orange')

  ax2=axes[i].twinx()
  ax2.set_ylim(0,90)
  ax2.set_ylabel('Separation of Barycenters')

  g = sns.lineplot(
    data=df1,
    x='t_mean', y='sar',color='black',
    estimator=estim, errorbar='sd', err_style='bars',
    marker='o', linewidth=2, markersize=6, alpha=0.4,
    ax=ax2,label=r'D [Mm]'
    )
  if i<2:
    ax2.set_ylabel('')
    ax2.set_yticklabels('')



# Etiquetas globales
for ax in axes.flat:
    ax.legend().remove()
    ax.axhline(0, linestyle='--', color='gray', alpha=0.6)
    ax.set_xlim(0,1)
    ax.set_ylim(-1,1)
    ax.set_xlabel(r'$t_{norm}$')
    ax.set_ylabel(r'$\gamma$ [deg/hour]')
axes[0].legend(loc='lower center',ncol=2)

plt.tight_layout()
plt.show()

# --- Nueva celda ---

DFx2.groupby(['AR','t_mean']).mean().reset_index()

# --- Nueva celda ---

sns.histplot(data=DFx2.groupby(['AR','t_mean']).mean().reset_index(),x='sarn',bins=10)
plt.axvline(np.median(DFx2.sarn),color='red')

plt.xlabel('Separation of Barycenters [Mm]')
plt.ylabel('ARs')

plt.show()

# --- Nueva celda ---

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

estim = 'median'

lat_ranges = [(0,39)]   # 👈 tus tres rangos de latitud

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)



DFx=DFx.assign(rot= lambda x: np.gradient(x.alpha,x.mag)/1.5)
DFx=DFx.assign(rotb= lambda x: np.gradient(x.alphab,x.mag)/1.5)
DFx2 = DFx.assign(frac=lambda x: np.abs(x.alpha - x.alphab))
DFx2 = DFx2.assign(t_mean=lambda x: round(10 * x.t_norm) / 10)

DFmax_alpha = (
    DFx2[DFx2.t_mean == 1]
    .groupby("AR")[["sar", "sepax"]]
    .mean()
    .reset_index()
    .rename(columns={"sar": "sar_max", "sepax": "sepax_max"})
)
DFx2 = DFx2.merge(DFmax_alpha, on="AR", how="left")

# Filtrar rango de latitudes

DFx2 = DFx2[DFx2.t_mean <= 1]
DFx2 = DFx2.assign(sarn=lambda x: x.sar /x.sar_max)
DFx2 = DFx2.assign(sarn=lambda x: round(10 * x.sarn) / 10)



    # ARs de alto y bajo twist
DFmax = DFx2[DFx2.t_mean == 1].groupby('AR').mean().reset_index()
ARup = DFmax[DFmax.flux >= np.median(DFmax.flux)].AR.unique()
ARlow = DFmax[DFmax.flux < np.median(DFmax.flux)].AR.unique()

ARpos1=DFmax[DFmax.rot_cum >0].AR.unique()
ARneg1=DFmax[DFmax.rot_cum <0].AR.unique()

ARpos2=DFmax[DFmax.rotb_cum >0].AR.unique()
ARneg2=DFmax[DFmax.rotb_cum <0].AR.unique()


for i, ars in enumerate([DFx.AR.unique(),ARup,ARlow]):



  df1=DFx2[DFx2.AR.isin(ars)]

  g = sns.lineplot(
    data=df1[df1.AR.isin(ARpos1)],
    x='sarn', y='rot',color='tab:blue',
    estimator=estim, errorbar='ci', err_style='bars',
    marker='o', linewidth=2, markersize=6, alpha=0.6,
    ax=axes[i],label=r'$\gamma_\mathrm{mod}>0$'
)
  g = sns.lineplot(
  data=df1[df1.AR.isin(ARneg1)],
  x='sarn', y='rot',color='tab:blue',
  estimator=estim, errorbar='ci', err_style='bars',
  marker='^', linewidth=2,linestyle='dashed', markersize=6, alpha=0.6,
  ax=axes[i],label=r'$\gamma_\mathrm{mod}$<0'
)

  g = sns.lineplot(
    data=df1[df1.AR.isin(ARpos2)],
    x='sarn', y='rotb',color='tab:orange',
    estimator=estim, errorbar='ci', err_style='bars',
    marker='o', linewidth=2, markersize=6, alpha=0.6,
    ax=axes[i],label=r'$\gamma_\mathrm{bar}>0$'
)
  g = sns.lineplot(
  data=df1[df1.AR.isin(ARneg2)],
  x='sarn', y='rotb',color='tab:orange',
  estimator=estim, errorbar='ci', err_style='bars',
  marker='^', linewidth=2,linestyle='dashed', markersize=6, alpha=0.6,
  ax=axes[i],label=r'$\gamma_\mathrm{bar}<0$'
)


  axes[i].set_title(f"{['All','High flux','Low flux'][i]} ARs")

  axes[i].text(0.7,0.5,f'{len(df1[df1.AR.isin(ARpos1)].AR.unique())}',color='tab:blue')
  axes[i].text(0.8,0.5,f'{len(df1[df1.AR.isin(ARpos2)].AR.unique())}',color='tab:orange')

  axes[i].text(0.7,-0.5,f'{len(df1[df1.AR.isin(ARneg1)].AR.unique())}',color='tab:blue')
  axes[i].text(0.8,-0.5,f'{len(df1[df1.AR.isin(ARneg2)].AR.unique())}',color='tab:orange')




# Etiquetas globales
for ax in axes.flat:
    ax.legend().remove()
    ax.axhline(0, linestyle='--', color='gray', alpha=0.6)
    ax.set_xlim(0,1)
    ax.set_ylim(-2,2)
    ax.set_xlabel(r'$D/D_{t=1}$')
    ax.set_ylabel(r'$\gamma$ [deg/hour]')
axes[0].legend(loc='lower center',ncol=2)

plt.tight_layout()
plt.show()

# --- Nueva celda ---

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

estim = 'median'

lat_ranges = [(0,39)]   # 👈 tus tres rangos de latitud

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)



DFx=DFx.assign(rot= lambda x: np.gradient(x.alpha,x.mag)/1.5)
DFx=DFx.assign(rotb= lambda x: np.gradient(x.alphab,x.mag)/1.5)
DFx2 = DFx.assign(frac=lambda x: np.abs(x.alpha - x.alphab))
DFx2 = DFx2.assign(t_mean=lambda x: round(10 * x.t_norm) / 10)

DFmax_alpha = (
    DFx2[DFx2.t_mean == 1]
    .groupby("AR")[["sar", "sepax"]]
    .mean()
    .reset_index()
    .rename(columns={"sar": "sar_max", "sepax": "sepax_max"})
)
DFx2 = DFx2.merge(DFmax_alpha, on="AR", how="left")

# Filtrar rango de latitudes

DFx2 = DFx2[DFx2.t_mean <= 1]
DFx2 = DFx2.assign(sarn=lambda x: x.sepax /x.sepax_max)
DFx2 = DFx2.assign(sarn=lambda x: round(10 * x.sarn) / 10)



    # ARs de alto y bajo twist
DFmax = DFx2[DFx2.t_mean == 1].groupby('AR').mean().reset_index()
ARup = DFmax[DFmax.flux >= np.median(DFmax.flux)].AR.unique()
ARlow = DFmax[DFmax.flux < np.median(DFmax.flux)].AR.unique()

ARpos1=DFmax[DFmax.rot_cum >0].AR.unique()
ARneg1=DFmax[DFmax.rot_cum <0].AR.unique()

ARpos2=DFmax[DFmax.rotb_cum >0].AR.unique()
ARneg2=DFmax[DFmax.rotb_cum <0].AR.unique()


for i, ars in enumerate([DFx.AR.unique(),ARup,ARlow]):



  df1=DFx2[DFx2.AR.isin(ars)]

  g = sns.lineplot(
    data=df1[df1.AR.isin(ARpos1)],
    x='sarn', y='rot',color='tab:blue',
    estimator=estim, errorbar='ci', err_style='bars',
    marker='o', linewidth=2, markersize=6, alpha=0.6,
    ax=axes[i],label=r'$\gamma_\mathrm{mod}>0$'
)
  g = sns.lineplot(
  data=df1[df1.AR.isin(ARneg1)],
  x='sarn', y='rot',color='tab:blue',
  estimator=estim, errorbar='ci', err_style='bars',
  marker='^', linewidth=2,linestyle='dashed', markersize=6, alpha=0.6,
  ax=axes[i],label=r'$\gamma_\mathrm{mod}$<0'
)

  g = sns.lineplot(
    data=df1[df1.AR.isin(ARpos2)],
    x='sarn', y='rotb',color='tab:orange',
    estimator=estim, errorbar='ci', err_style='bars',
    marker='o', linewidth=2, markersize=6, alpha=0.6,
    ax=axes[i],label=r'$\gamma_\mathrm{bar}>0$'
)
  g = sns.lineplot(
  data=df1[df1.AR.isin(ARneg2)],
  x='sarn', y='rotb',color='tab:orange',
  estimator=estim, errorbar='ci', err_style='bars',
  marker='^', linewidth=2,linestyle='dashed', markersize=6, alpha=0.6,
  ax=axes[i],label=r'$\gamma_\mathrm{bar}<0$'
)


  axes[i].set_title(f"{['All','High flux','Low flux'][i]} ARs")

  axes[i].text(0.7,0.5,f'{len(df1[df1.AR.isin(ARpos1)].AR.unique())}',color='tab:blue')
  axes[i].text(0.8,0.5,f'{len(df1[df1.AR.isin(ARpos2)].AR.unique())}',color='tab:orange')

  axes[i].text(0.7,-0.5,f'{len(df1[df1.AR.isin(ARneg1)].AR.unique())}',color='tab:blue')
  axes[i].text(0.8,-0.5,f'{len(df1[df1.AR.isin(ARneg2)].AR.unique())}',color='tab:orange')




# Etiquetas globales
for ax in axes.flat:
    ax.legend().remove()
    ax.axhline(0, linestyle='--', color='gray', alpha=0.6)
    ax.set_xlim(0,1)
    ax.set_ylim(-1,1)
    ax.set_xlabel(r'$D/D_{t=1}$')
    ax.set_ylabel(r'$\gamma$ [deg/hour]')
axes[0].legend(loc='lower center',ncol=2)

plt.tight_layout()
plt.show()

# --- Nueva celda ---

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

estim = 'median'

lat_ranges = [(0,39)]   # 👈 tus tres rangos de latitud

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

DFx=DFx.assign(rot= lambda x: np.gradient(x.alpha,x.mag)/1.5)
DFx=DFx.assign(rotb= lambda x: np.gradient(x.alphab,x.mag)/1.5)


# Filtrar rango de latitudes
DFx2 = DFx.assign(frac=lambda x: np.abs(x.alpha - x.alphab))
DFx2 = DFx2.assign(t_mean=lambda x: round(10 * x.t_norm) / 10)
DFx2 = DFx2[DFx2.t_mean <= 1]
DFx2 = DFx2.assign(tw=lambda x: np.abs(x.Nt) * x.flux**2)
    # ARs de alto y bajo twist
DFmax = DFx2[DFx2.t_mean == 1].groupby('AR').mean().reset_index()
ARup = DFmax[DFmax.flux >= np.median(DFmax.flux)].AR.unique()
ARlow = DFmax[DFmax.flux < np.median(DFmax.flux)].AR.unique()

ARpos1=DFmax[DFmax.rot_cum >0].AR.unique()
ARneg1=DFmax[DFmax.rot_cum <0].AR.unique()

ARpos2=DFmax[DFmax.rotb_cum >0].AR.unique()
ARneg2=DFmax[DFmax.rotb_cum <0].AR.unique()


for i, ars in enumerate([DFx.AR.unique(),ARup,ARlow]):



  df1=DFx2[DFx2.AR.isin(ars)]

  g = sns.lineplot(
    data=df1[df1.AR.isin(ARpos1)],
    x='t_mean', y='rot',color='tab:blue',
    estimator=estim, errorbar='ci', err_style='bars',
    marker='o', linewidth=2, markersize=6, alpha=0.6,
    ax=axes[i],label=r'$\gamma_\mathrm{mod}>0$'
)
  g = sns.lineplot(
  data=df1[df1.AR.isin(ARneg1)],
  x='t_mean', y='rot',color='tab:blue',
  estimator=estim, errorbar='ci', err_style='bars',
  marker='^', linewidth=2,linestyle='dashed', markersize=6, alpha=0.6,
  ax=axes[i],label=r'$\gamma_\mathrm{mod}$<0'
)

  g = sns.lineplot(
    data=df1[df1.AR.isin(ARpos2)],
    x='t_mean', y='rotb',color='tab:orange',
    estimator=estim, errorbar='ci', err_style='bars',
    marker='o', linewidth=2, markersize=6, alpha=0.6,
    ax=axes[i],label=r'$\gamma_\mathrm{bar}>0$'
)
  g = sns.lineplot(
  data=df1[df1.AR.isin(ARneg2)],
  x='t_mean', y='rotb',color='tab:orange',
  estimator=estim, errorbar='ci', err_style='bars',
  marker='^', linewidth=2,linestyle='dashed', markersize=6, alpha=0.6,
  ax=axes[i],label=r'$\gamma_\mathrm{bar}<0$'
)


  axes[i].set_title(f"{['All','High flux','Low flux'][i]} ARs")

  axes[i].text(0.7,0.5,f'{len(df1[df1.AR.isin(ARpos1)].AR.unique())}',color='tab:blue')
  axes[i].text(0.8,0.5,f'{len(df1[df1.AR.isin(ARpos2)].AR.unique())}',color='tab:orange')

  axes[i].text(0.7,-0.5,f'{len(df1[df1.AR.isin(ARneg1)].AR.unique())}',color='tab:blue')
  axes[i].text(0.8,-0.5,f'{len(df1[df1.AR.isin(ARneg2)].AR.unique())}',color='tab:orange')

  ax2=axes[i].twinx()
  ax2.set_ylim(0,90)
  ax2.set_ylabel('Axial Separation [Mm]')

  g = sns.lineplot(
    data=df1,
    x='t_mean', y='sepax',color='black',
    estimator=estim, errorbar='sd', err_style='bars',
    marker='o', linewidth=2, markersize=6, alpha=0.4,
    ax=ax2,label=r'D [Mm]'
    )
  if i<2:
    ax2.set_ylabel('')
    ax2.set_yticklabels('')


# Etiquetas globales
for ax in axes.flat:
    ax.legend().remove()
    ax.axhline(0, linestyle='--', color='gray', alpha=0.6)
    ax.set_xlim(0,1)
    ax.set_ylim(-1,1)
    ax.set_xlabel(r'$t_{norm}$')
    ax.set_ylabel(r'$\gamma$ [deg/hour]')
axes[0].legend(loc='lower center',ncol=2)

plt.tight_layout()
plt.show()

# --- Nueva celda ---

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

estim = 'median'

lat_ranges = [(0,39)]   # 👈 tus tres rangos de latitud

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

DFx=DFx.assign(rot= lambda x: np.gradient(x.alpha,x.mag)/1.5)
DFx=DFx.assign(rotb= lambda x: np.gradient(x.alphab,x.mag)/1.5)




# Filtrar rango de latitudes
DFx2 = DFx.assign(frac=lambda x: np.abs(x.alpha - x.alphab))
DFx2 = DFx2.assign(t_mean=lambda x: round(10 * x.t_norm) / 10)
DFx2 = DFx2[DFx2.t_mean <= 1]
DFx2 = DFx2.assign(tw=lambda x: np.abs(x.Nt) * x.flux**2)
    # ARs de alto y bajo twist
# --- tomar alpha y alphab al máximo (t_mean=1) ---
DFmax_alpha = (
    DFx2[DFx2.t_mean == 1]
    .groupby("AR")[["alpha", "alphab"]]
    .mean()
    .reset_index()
    .rename(columns={"alpha": "alpha_max", "alphab": "alphab_max"})
)

DFx2 = DFx2.merge(DFmax_alpha, on="AR", how="left")
DFmax = DFx2[DFx2.t_mean == 1].groupby('AR').mean().reset_index()
DFx2=DFx2.assign(fraca=lambda x: x.rot/x.alpha)
DFx2=DFx2.assign(fracb=lambda x: x.rotb/x.alphab)
ARup = DFmax[DFmax.flux >= np.median(DFmax.flux)].AR.unique()
ARlow = DFmax[DFmax.flux < np.median(DFmax.flux)].AR.unique()

ARpos1=DFmax[DFmax.rot_cum/DFmax.alpha >0].AR.unique()
ARneg1=DFmax[DFmax.rot_cum/DFmax.alpha <0].AR.unique()

ARpos2=DFmax[DFmax.rotb_cum/DFmax.alphab >0].AR.unique()
ARneg2=DFmax[DFmax.rotb_cum/DFmax.alphab <0].AR.unique()


for i, ars in enumerate([DFx.AR.unique(),ARup,ARlow]):



  df1=DFx2[DFx2.AR.isin(ars)]

  g = sns.lineplot(
    data=df1[df1.AR.isin(ARpos1)],
    x='t_mean', y='fraca',color='tab:blue',
    estimator=estim, errorbar='ci', err_style='bars',
    marker='o', linewidth=2, markersize=6, alpha=0.6,
    ax=axes[i],label=r'$\gamma_\mathrm{mod}/\alpha_\mathrm{mod}>0$'
)
  g = sns.lineplot(
  data=df1[df1.AR.isin(ARneg1)],
  x='t_mean', y='fraca',color='tab:blue',
  estimator=estim, errorbar='ci', err_style='bars',
  marker='^', linewidth=2,linestyle='dashed', markersize=6, alpha=0.6,
  ax=axes[i],label=r'$\gamma_\mathrm{mod}/\alpha_\mathrm{mod}$<0'
)

  g = sns.lineplot(
    data=df1[df1.AR.isin(ARpos2)],
    x='t_mean', y='fracb',color='tab:orange',
    estimator=estim, errorbar='ci', err_style='bars',
    marker='o', linewidth=2, markersize=6, alpha=0.6,
    ax=axes[i],label=r'$\gamma_\mathrm{bar}/\alpha_\mathrm{bar}>0$'
)
  g = sns.lineplot(
  data=df1[df1.AR.isin(ARneg2)],
  x='t_mean', y='fracb',color='tab:orange',
  estimator=estim, errorbar='ci', err_style='bars',
  marker='^', linewidth=2,linestyle='dashed', markersize=6, alpha=0.6,
  ax=axes[i],label=r'$\gamma_\mathrm{bar}/\alpha_\mathrm{bar}<0$'
)


  axes[i].set_title(f"{['All','High flux','Low flux'][i]} ARs")

  axes[i].text(0.7,0.04,f'{len(df1[df1.AR.isin(ARpos1)].AR.unique())}',color='tab:blue')
  axes[i].text(0.8,0.04,f'{len(df1[df1.AR.isin(ARpos2)].AR.unique())}',color='tab:orange')

  axes[i].text(0.7,-0.04,f'{len(df1[df1.AR.isin(ARneg1)].AR.unique())}',color='tab:blue')
  axes[i].text(0.8,-0.04,f'{len(df1[df1.AR.isin(ARneg2)].AR.unique())}',color='tab:orange')


# Etiquetas globales
for ax in axes.flat:
    ax.legend().remove()
    ax.axhline(0, linestyle='--', color='gray', alpha=0.6)
    ax.set_xlim(0,1)
    ax.set_ylim(-0.1,0.05)
    ax.set_xlabel(r'$t_{norm}$')
    ax.set_ylabel(r'$\gamma/\alpha$ [1/hour]')
axes[0].legend(loc='lower right')

plt.tight_layout()
plt.show()

# --- Nueva celda ---

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

estim = 'median'

lat_ranges = [(0,39)]   # 👈 tus tres rangos de latitud

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

DFx=DFx.assign(rot= lambda x: np.gradient(x.alpha,x.mag)/1.5)
DFx=DFx.assign(rotb= lambda x: np.gradient(x.alphab,x.mag)/1.5)




# Filtrar rango de latitudes
DFx2 = DFx.assign(frac=lambda x: np.abs(x.alpha - x.alphab))
DFx2 = DFx2.assign(t_mean=lambda x: round(10 * x.t_norm) / 10)
DFx2 = DFx2[DFx2.t_mean <= 1]
DFx2 = DFx2.assign(tw=lambda x: np.abs(x.Nt) * x.flux**2)
    # ARs de alto y bajo twist
# --- tomar alpha y alphab al máximo (t_mean=1) ---
DFmax_alpha = (
    DFx2[DFx2.t_mean == 1]
    .groupby("AR")[["alpha", "alphab"]]
    .mean()
    .reset_index()
    .rename(columns={"alpha": "alpha_max", "alphab": "alphab_max"})
)

DFx2 = DFx2.merge(DFmax_alpha, on="AR", how="left")
DFmax = DFx2[DFx2.t_mean == 1].groupby('AR').mean().reset_index()
DFx2=DFx2.assign(fraca=lambda x: np.sign(x.alpha)*x.rot)
DFx2=DFx2.assign(fracb=lambda x: np.sign(x.alpha)*x.rotb)
ARup = DFmax[DFmax.flux >= np.median(DFmax.flux)].AR.unique()
ARlow = DFmax[DFmax.flux < np.median(DFmax.flux)].AR.unique()

ARpos1=DFmax[DFmax.rot_cum/DFmax.alpha >0].AR.unique()
ARneg1=DFmax[DFmax.rot_cum/DFmax.alpha <0].AR.unique()

ARpos2=DFmax[DFmax.rotb_cum/DFmax.alphab >0].AR.unique()
ARneg2=DFmax[DFmax.rotb_cum/DFmax.alphab <0].AR.unique()


for i, ars in enumerate([DFx.AR.unique(),ARup,ARlow]):



  df1=DFx2[DFx2.AR.isin(ars)]

  g = sns.lineplot(
    data=df1[df1.AR.isin(ARpos1)],
    x='t_mean', y='fraca',color='tab:blue',
    estimator=estim, errorbar='ci', err_style='bars',
    marker='o', linewidth=2, markersize=6, alpha=0.6,
    ax=axes[i],label=r'$\gamma_\mathrm{mod}>0$'
)
  g = sns.lineplot(
  data=df1[df1.AR.isin(ARneg1)],
  x='t_mean', y='fraca',color='tab:blue',
  estimator=estim, errorbar='ci', err_style='bars',
  marker='^', linewidth=2,linestyle='dashed', markersize=6, alpha=0.6,
  ax=axes[i],label=r'$\gamma_\mathrm{mod}<0'
)

  g = sns.lineplot(
    data=df1[df1.AR.isin(ARpos2)],
    x='t_mean', y='fracb',color='tab:orange',
    estimator=estim, errorbar='ci', err_style='bars',
    marker='o', linewidth=2, markersize=6, alpha=0.6,
    ax=axes[i],label=r'$\gamma_\mathrm{bar}>0$'
)
  g = sns.lineplot(
  data=df1[df1.AR.isin(ARneg2)],
  x='t_mean', y='fracb',color='tab:orange',
  estimator=estim, errorbar='ci', err_style='bars',
  marker='^', linewidth=2,linestyle='dashed', markersize=6, alpha=0.6,
  ax=axes[i],label=r'$\gamma_\mathrm{bar}<0$'
)


  axes[i].set_title(f"{['All','High flux','Low flux'][i]} ARs")

  axes[i].text(0.7,0.04,f'{len(df1[df1.AR.isin(ARpos1)].AR.unique())}',color='tab:blue')
  axes[i].text(0.8,0.04,f'{len(df1[df1.AR.isin(ARpos2)].AR.unique())}',color='tab:orange')

  axes[i].text(0.7,-0.04,f'{len(df1[df1.AR.isin(ARneg1)].AR.unique())}',color='tab:blue')
  axes[i].text(0.8,-0.04,f'{len(df1[df1.AR.isin(ARneg2)].AR.unique())}',color='tab:orange')


# Etiquetas globales
for ax in axes.flat:
    ax.legend().remove()
    ax.axhline(0, linestyle='--', color='gray', alpha=0.6)
    ax.set_xlim(0,1)
   # ax.set_ylim(-0.1,0.05)
    ax.set_xlabel(r'$t_{norm}$')
    ax.set_ylabel(r'$\gamma$ [deg/hour]')
axes[0].legend()

plt.tight_layout()
plt.show()

# --- Nueva celda ---

sns.histplot(data=DFrot,x='rot_cum', binrange=(-50,50),bins=10)
sns.histplot(data=DFrot,x='rotb_cum', binrange=(-50,50),bins=10)

plt.xlabel('Cumulative rotation [deg]')
plt.ylabel('ARs')
plt.text(-48,25,f'$\mu$={DFrot.rot_cum.mean():.2f}\n$\sigma$={DFrot.rot_cum.std():.2f}\nskew ={DFrot.rot_cum.skew():.2f}',color='tab:blue')
plt.text(20,25,f'$\mu$={DFrot.rotb_cum.mean():.2f}\n$\sigma$={DFrot.rotb_cum.std():.2f}\nskew={DFrot.rotb_cum.skew():.2f}',color='tab:orange')

plt.show()

sns.histplot(data=DFrot.apply(lambda x: np.sign(DFrot.alpha)*x),x='rot_cum', binrange=(-50,50),bins=10)
sns.histplot(data=DFrot.apply(lambda x: np.sign(DFrot.alphab)*x),x='rotb_cum', binrange=(-50,50),bins=10)

a=DFrot.apply(lambda x: np.sign(DFrot.alpha)*x)['rot_cum']
b=DFrot.apply(lambda x: np.sign(DFrot.alphab)*x)['rotb_cum']

plt.text(-48,25,f'$\mu$={a.mean():.2f}\n$\sigma$={a.std():.2f}\nskew ={a.skew():.2f}',color='tab:blue')
plt.text(20,25,f'$\mu$={b.mean():.2f}\n$\sigma$={b.std():.2f}\nskew={b.skew():.2f}',color='tab:orange')

plt.xlabel(r'Sign of $\alpha$ * Cumulative rotation [deg]')
plt.ylabel('ARs')


plt.show()


# --- Nueva celda ---


DFrot.apply(lambda x: np.sign(DFrot.alpha)*x)

# --- Nueva celda ---

fig, axes = plt.subplots(1, 2, figsize=(12,5), sharex=True, sharey=True)


DFrot=DFrot.assign(rot_cum=lambda x: np.sign(x.lat)*x.rot_cum)
DFrot=DFrot.assign(rotb_cum=lambda x: np.sign(x.lat)*x.rotb_cum)
# --- Case Nt > 0 ---
sns.histplot(
    data=pd.melt(DFrot[DFrot.Nt > 0], value_vars=['rot_cum','rotb_cum'],
                 value_name='acum', var_name='var'),
    x='acum', hue='var', binrange=(-50,50), ax=axes[0]
)

axes[0].set_title("Nt > 0")
axes[0].text(-20, 10,
             fr'$\mu$={DFrot[DFrot.Nt > 0].rot_cum.mean():.2f}, '
             fr'$\sigma$={DFrot[DFrot.Nt > 0].rot_cum.std():.2f}, '
             fr'skew={DFrot[DFrot.Nt > 0].rot_cum.skew():.2f}',
             color='tab:blue')
axes[0].text(-20, 9,
             fr'$\mu$={DFrot[DFrot.Nt > 0].rotb_cum.mean():.2f}, '
             fr'$\sigma$={DFrot[DFrot.Nt > 0].rotb_cum.std():.2f}, '
             fr'skew={DFrot[DFrot.Nt > 0].rotb_cum.skew():.2f}',
             color='tab:orange')

# --- Case Nt < 0 ---
sns.histplot(
    data=pd.melt(DFrot[DFrot.Nt < 0], value_vars=['rot_cum','rotb_cum'],
                 value_name='acum', var_name='var'),
    x='acum', hue='var', binrange=(-50,50), ax=axes[1]
)

axes[1].set_title("Nt < 0")
axes[1].text(-20, 10,
             fr'$\mu$={DFrot[DFrot.Nt < 0].rot_cum.mean():.2f}, '
             fr'$\sigma$={DFrot[DFrot.Nt < 0].rot_cum.std():.2f}, '
             fr'skew={DFrot[DFrot.Nt < 0].rot_cum.skew():.2f}',
             color='tab:blue')
axes[1].text(-20, 9,
             fr'$\mu$={DFrot[DFrot.Nt < 0].rotb_cum.mean():.2f}, '
             fr'$\sigma$={DFrot[DFrot.Nt < 0].rotb_cum.std():.2f}, '
             fr'skew={DFrot[DFrot.Nt < 0].rotb_cum.skew():.2f}',
             color='tab:orange')

plt.tight_layout()
plt.show()

# --- Nueva celda ---

#  Jointplot de seaborn con histograma extendido en marg_y
g = sns.jointplot(
    data=pd.melt(DFrot,id_vars=['t_norm','lat','AR','fn','mag','flux','Nt'],value_vars=['rot_cum','rotb_cum']),
    x='Nt',
    y='value',
    hue='variable',
    kind='scatter',
    alpha=0.7,
 #   marginal_kws={'bins': 30,'binrange':(-80,80), 'fill': True},
    ratio=2,height=6,space=0.3
)

g.set_axis_labels(r'$N_\mathrm{t}$',r'$\gamma$ [deg]', fontsize=12)

# Linear fit
slope, intercept, r_value, p_value, std_err = stats.linregress(DFrot['Nt'], DFrot['rot_cum'])

print(f'r = {r_value}')

slope, intercept, r_value, p_value, std_err = stats.linregress(DFrot['Nt'], DFrot['rotb_cum'])

print(f'r = {r_value}')


# --- Nueva celda ---

DFrot=DFrot.assign(tw=lambda x: x.Nt*x.flux**2)

#  Jointplot de seaborn con histograma extendido en marg_y
g = sns.jointplot(
    data=pd.melt(DFrot,id_vars=['t_norm','lat','AR','fn','mag','flux','Nt','tw'],value_vars=['rot_cum','rotb_cum']),
    x='tw',
    y='value',
    hue='variable',
    kind='scatter',
    alpha=0.7,
 #   marginal_kws={'bins': 30,'binrange':(-80,80), 'fill': True},
    ratio=2,height=6,space=0.3
)

g.set_axis_labels(r'$N_\mathrm{t}$',r'$\gamma$ [deg]', fontsize=12)

# Linear fit
slope, intercept, r_value, p_value, std_err = stats.linregress(DFrot['tw'], DFrot['rot_cum'])

print(f'r = {r_value}')

slope, intercept, r_value, p_value, std_err = stats.linregress(DFrot['tw'], DFrot['rotb_cum'])

print(f'r = {r_value}')


# --- Nueva celda ---

#  Jointplot de seaborn con histograma extendido en marg_y

DFrot = DFrot[~DFrot['AR'].isin(ARout)].copy()

g = sns.jointplot(
    data=DFrot,
    x='rotb_cum',
    y='rot_cum',
    kind='scatter',
    alpha=0.7,
    marginal_kws={'bins': 30,'binrange':(-80,80), 'fill': True},
    ratio=2,height=6,space=0.3
)

g.set_axis_labels(r'$\gamma_\mathrm{bar}$ [deg/hour]',r'$\gamma_\mathrm{mod}$ [deg/hour]', fontsize=12)

g.ax_marg_x.set_visible(False)
#g.ax_marg_y.set_visible(False)
g.ax_marg_x.clear()
g.ax_marg_y.clear()
g.ax_marg_x.set_xlabel('X axis label')
g.ax_marg_y.set_ylabel('Y axis label')
g.ax_marg_x.tick_params(axis='x', labelsize=12)
g.ax_marg_y.tick_params(axis='y', labelsize=12)

hist=sns.histplot(data=pd.melt(DFrot,id_vars=['t_norm','lat','AR','fn','mag','flux'],value_vars=['rot_cum','rotb_cum']),y='value',
                  hue='variable',
             palette=['tab:blue','tab:orange'],
             alpha=0.5,ax=g.ax_marg_y,binrange=(-60,60),bins=40)

# Linear fit
slope, intercept, r_value, p_value, std_err = stats.linregress(DFrot['rotb_cum'], DFrot['rot_cum'])

# Add linear fit line
x_fit = np.linspace(DFrot['rotb_cum'].min(), DFrot['rotb_cum'].max(), 100)
y_fit = slope * x_fit + intercept
g.ax_joint.plot(x_fit, y_fit, color='red', label=f'Fit: y={slope:.2f}x+{intercept:.2f}')

# Annotate with correlation coefficient and slope
g.ax_joint.text(
    0.65, 0.35,
    f'$r$ = {r_value:.2f}\nSlope = {slope:.2f}',
    transform=g.ax_joint.transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle="round", alpha=0.2, color='red')
)

#g.ax_marg_y.legend()
g.ax_marg_y.legend(title='',labels=[r'$\gamma_\mathrm{bar}$',r'$\gamma_\mathrm{mod}$'],loc='upper right')
handles,labels=hist.get_legend_handles_labels()
#hist.legend(handles=handles,labels=labels,title=r'$t_{norm}$')
handles,labels=hist.get_legend_handles_labels()
g.ax_marg_y.set_ylabel('')
#
g.ax_marg_y.tick_params(axis='y', labelsize=10)  # Set y-ticklabel fontsize
g.ax_marg_y.tick_params(axis='x', labelsize=10)  # Set x-ticklabel fontsize (if needed)

g.ax_marg_y.axhline(0,color='black')

g.ax_marg_y.axhline(DFrot.rot_cum.mean(),color='tab:blue')
g.ax_marg_y.axhline(DFrot.rotb_cum.mean(),color='tab:orange')

g.ax_joint.axvline(0,color='black',linestyle='dashed')
g.ax_joint.axhline(0,color='black',linestyle='dashed')
#g.ax_joint.set_ylim(-20, 20)
#g.ax_joint.set_xlim(-20, 20)

g.ax_marg_y.text(
    0.45, 0.15,
    f'$\sigma_\mathrm{{mod}}$ = {DFrot.rot_cum.std():.0f}\n$\sigma_\mathrm{{bar}}$ = {DFrot.rotb_cum.std():.0f}',
    transform=g.ax_marg_y.transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle="round", alpha=0.0, color='red')
)

#g.ax_marg_y.set_xlim(0,20)
'''
g.ax_joint.plot([-20,20],[20,20],color='black')
g.ax_joint.plot([20,20],[-20,20],color='black')
g.ax_marg_y.plot([0,20],[-20,-20],color='black')
g.ax_marg_y.plot([0,20],[20,20],color='black')
g.ax_marg_y.plot([20,20],[-20,20],color='black')
'''

aa=DFrot.rot_cum.values
ab=DFrot.rotb_cum.values



print(round(100*sum((aa>0)*(ab>0))/len(aa)))
print(round(100*sum((aa<0)*(ab<0))/len(aa)))
print(round(100*sum((aa>0)*(ab<0))/len(aa)))
print(round(100*sum((aa<0)*(ab>0))/len(aa)))

g.ax_joint.text(0.7,0.9,f'~{round(100*sum((aa>0)*(ab>0))/len(aa))}%',transform=g.ax_joint.transAxes,
                fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle="round", alpha=0.5, color='tab:blue'))
g.ax_joint.text(0.15,0.9,f'~{round(100*sum((aa>0)*(ab<0))/len(aa))}%',transform=g.ax_joint.transAxes,
                fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle="round", alpha=0.5, color='tab:blue'))
g.ax_joint.text(0.7,0.2,f'~{round(100*sum((aa<0)*(ab>0))/len(aa))}%',transform=g.ax_joint.transAxes,
                fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle="round", alpha=0.5, color='tab:blue'))
g.ax_joint.text(0.15,0.2,f'~{round(100*sum((aa<0)*(ab<0))/len(aa))}%',transform=g.ax_joint.transAxes,
                fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle="round", alpha=0.5, color='tab:blue'))
#plt.savefig('./plots/alpha-alpha.pdf',dpi=300)

# --- Nueva celda ---

f=plt.figure(figsize=(6,3))

DFx2=DFx.assign(frac=lambda x: np.abs(x.alpha-x.alphab))
DFx2=DFx2.assign(t_mean=lambda x: round(10*x.t_norm)/10)

g=sns.lineplot(
data=DFx2,
x='t_mean',
y='frac',
estimator='mean',
errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='bars',   # o 'bars' para barras verticales
marker='o',          # opcional para marcar puntos medios
linewidth=3,markersize=10,alpha=0.5, label='Diferencia de tilt'

                )
#g=sns.scatterplot(data=DFx2,x='t_mean',y='frac',alpha=0.2)
#   sns.lineplot(data=DFx2.groupby(['t_norm','variable']).median().reset_index(),x='t_norm',y='value',hue='variable')
plt.xlim(0,1)
plt.ylim(-5,40)
plt.axhline(0,color='black',linestyle='dashed')

plt.ylabel(r'$|\alpha_{mod}-\alpha_{bar}|$  [deg]',fontsize=12)
plt.xlabel(r'$t_{norm}$',fontsize=12)
#plt.set_ticklabel(size=12)
plt.tick_params(axis='both', labelsize=12)
plt.legend(loc='lower left')

ax2=plt.twinx()
g=sns.lineplot(
data=DFx2,
x='t_mean',
y='fn',
color='tab:orange',
estimator='mean',
errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='bars',   # o 'bars' para barras verticales
marker='^',          # opcional para marcar puntos medios
linewidth=2,markersize=10,ax=ax2,alpha=0.5,label='Flujo Normalizado')
ax2.set_ylabel('Flujo Normalizado',fontsize=12)
ax2.tick_params(axis='y', labelsize=12)

ax2.set_ylim(0,1)
plt.legend(loc='lower right')

plt.show()




# --- Nueva celda ---

DFx=DFx.assign(rot= lambda x: np.gradient(x.alpha,x.mag)/1.5)
DFx=DFx.assign(rotb= lambda x: np.gradient(x.alphab,x.mag)/1.5)
DFx["rot_cum"]  = DFx.groupby("AR")["rot"].cumsum()
DFx["rotb_cum"] = DFx.groupby("AR")["rotb"].cumsum()


DFx2=DFx.assign(frac=lambda x: np.abs(x.alpha-x.alphab))
DFx2=DFx2.assign(t_mean=lambda x: round(10*x.t_norm)/10)


#DFx2=DFx2.groupby(['AR','t_mean']).max().reset_index()

# --- Nueva celda ---

DFx2=DFx.assign(frac=lambda x: np.abs(x.alpha-x.alphab))
DFx2=DFx2.assign(t_mean=lambda x: round(10*x.t_norm)/10)

# --- Nueva celda ---

pd.melt(DFx2,id_vars=['AR','t_mean'],value_vars=['rot','rotb'],var_name='variable')

# --- Nueva celda ---

DFx2=DFx.assign(fraca=lambda x: np.abs(x.alpha-x.alphab))

DFx2=DFx2.assign(t_mean=lambda x: round(10*x.t_norm)/10)

DFx2=DFx2.assign(rot_cum=lambda x: x.rot_cum*1.5*x.stp)
DFx2=DFx2.assign(rotb_cum=lambda x: x.rotb_cum*1.5*x.stp)

f=plt.figure(figsize=(6,3))


g=sns.lineplot(
data=pd.melt(np.abs(DFx2),id_vars=['AR','t_mean'],value_vars=['rot','rotb'],var_name='variable'),
x='t_mean',
y='value',
estimator='median',
                hue='variable',
errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='bars',   # o 'bars' para barras verticales
marker='o',          # opcional para marcar puntos medios
#color='tab:blue',
linewidth=3,markersize=10,alpha=0.5

                )




#g=sns.scatterplot(data=DFx2,x='t_mean',y='frac',alpha=0.2)
#   sns.lineplot(data=DFx2.groupby(['t_norm','variable']).median().reset_index(),x='t_norm',y='value',hue='variable')
plt.xlim(0,1)
plt.ylim(0,2)
plt.axhline(0,color='black',linestyle='dashed')

plt.ylabel(r'$|\Delta \alpha|$  [deg/hour]',fontsize=12)
plt.xlabel(r'$t_{norm}$',fontsize=12)
#plt.set_ticklabel(size=12)
plt.tick_params(axis='both', labelsize=12)
plt.legend(loc='upper right',fontsize=12)


ax2=plt.twinx()




g=sns.lineplot(
data=pd.melt(np.abs(DFx2),id_vars=['AR','t_mean'],value_vars=['rot_cum','rotb_cum'],var_name='variable'),
x='t_mean',
y='value',
                hue='variable',
estimator='median',
errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='band',   # o 'bars' para barras verticales
marker='^',          # opcional para marcar puntos medios
#color='tab:blue',
                ax=ax2,
linewidth=2,linestyle='dashed',markersize=10,alpha=0.5#, label=r'$\Delta \alpha_\mathrm{mod}$'

                )


ax2.set_ylim(-10,20)

ax2.tick_params(axis='both', labelsize=12)
ax2.set_ylabel(r'$\Delta\alpha$ Acumulado [deg/hour]',fontsize=12)
#plt.legend(loc='lower right')
ax2.legend().remove()

plt.show()




# --- Nueva celda ---

DFx2 = DFx.assign(fraca=lambda x: np.abs(x.alpha - x.alphab))
DFx2 = DFx2.assign(t_mean=lambda x: round(10 * x.t_norm) / 10)
DFx2 = DFx2.assign(rot_cum=lambda x: x.rot_cum * 1.5 * x.stp)
DFx2 = DFx2.assign(rotb_cum=lambda x: x.rotb_cum * 1.5 * x.stp)

DFmax = DFx2[DFx2.t_mean == 1].groupby('AR').mean().reset_index()
ARup = DFmax[DFmax.flux >= np.median(DFmax.flux)].AR.unique()
ARlow = DFmax[DFmax.flux < np.median(DFmax.flux)].AR.unique()

# --- 3 filas, 1 columna ---
fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True, sharey=True)

for i, ARs in enumerate([DFmax.AR.unique(), ARup, ARlow]):
    ax1 = axes[i]

    # --- rot y rotb ---
    df1 = pd.melt(
        DFx2,
        id_vars=['AR', 't_mean'],
        value_vars=['rot', 'rotb'],
        var_name='variable',
        value_name='value'
    )

    sns.lineplot(
        data=df1[df1.AR.isin(ARs)].assign(value=lambda d: d.value),
        x='t_mean',
        y='value',
        estimator='median',
        hue='variable',
        errorbar='ci',
        err_style='bars',
        marker='o',
        linewidth=3,
        markersize=10,
        alpha=0.5,
        ax=ax1
    )

    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.5, 0.5)
    ax1.axhline(0, color='black', linestyle='dashed')
    ax1.set_ylabel(r'$\gamma $ [deg/hour]', fontsize=12)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.legend(loc='upper right', fontsize=10)

    # --- Segundo eje: rot_cum y rotb_cum ---
    ax2 = ax1.twinx()

    df2 = pd.melt(
        DFx2,
        id_vars=['AR', 't_mean'],
        value_vars=['rot_cum', 'rotb_cum'],
        var_name='variable',
        value_name='value'
    )

    sns.lineplot(
        data=df2[df2.AR.isin(ARs)].assign(value=lambda d: d.value),
        x='t_mean',
        y='value',
        hue='variable',
        estimator='median',
        errorbar='ci',
        err_style='band',
        marker='^',
        linewidth=2,
        linestyle='dashed',
        markersize=10,
        alpha=0.5,
        ax=ax2
    )

    ax2.set_ylim(-10, 10)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_ylabel(r'Acummulated $\gamma$ [deg]', fontsize=12)
    ax2.legend().remove()

    # --- Títulos específicos ---
    if i == 0:
        ax1.set_title(f"All ARs ({len(ARs)})", fontsize=14)
    elif i == 1:
        ax1.set_title(f"High flux ARs ({len(ARs)})", fontsize=14)
    else:
        ax1.set_title(f"Low flux ARs ({len(ARs)})", fontsize=14)

# Etiqueta común para el eje x
axes[-1].set_xlabel(r'$t_{norm}$', fontsize=12)

plt.tight_layout()
plt.show()



# --- Nueva celda ---

DFx2 = DFx.assign(fraca=lambda x: np.abs(x.alpha - x.alphab))
DFx2 = DFx2.assign(t_mean=lambda x: round(10 * x.t_norm) / 10)
DFx2 = DFx2.assign(rot_cum=lambda x: x.rot_cum * 1.5 * x.stp)
DFx2 = DFx2.assign(rotb_cum=lambda x: x.rotb_cum * 1.5 * x.stp)

DFmax = DFx2[DFx2.t_mean == 1].groupby('AR').mean().reset_index()
DFmax  =  DFmax.assign(tw= lambda x: np.abs(x.Nt)*x.flux**2)
ARup = DFmax[DFmax.tw >= np.median(DFmax.tw)].AR.unique()
ARlow = DFmax[DFmax.tw < np.median(DFmax.tw)].AR.unique()

# --- 3 filas, 1 columna ---
fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True, sharey=True)

for i, ARs in enumerate([DFmax.AR.unique(), ARup, ARlow]):
    ax1 = axes[i]

    # --- rot y rotb ---
    df1 = pd.melt(
        DFx2,
        id_vars=['AR', 't_mean'],
        value_vars=['rot', 'rotb'],
        var_name='variable',
        value_name='value'
    )

    sns.lineplot(
        data=df1[df1.AR.isin(ARs)].assign(value=lambda d: d.value),
        x='t_mean',
        y='value',
        estimator='median',
        hue='variable',
        errorbar='ci',
        err_style='bars',
        marker='o',
        linewidth=3,
        markersize=10,
        alpha=0.5,
        ax=ax1
    )

    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.5, 0.5)
    ax1.axhline(0, color='black', linestyle='dashed')
    ax1.set_ylabel(r'$\gamma $ [deg/hour]', fontsize=12)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.legend(loc='upper right', fontsize=10)

    # --- Segundo eje: rot_cum y rotb_cum ---
    ax2 = ax1.twinx()

    df2 = pd.melt(
        DFx2,
        id_vars=['AR', 't_mean'],
        value_vars=['rot_cum', 'rotb_cum'],
        var_name='variable',
        value_name='value'
    )

    sns.lineplot(
        data=df2[df2.AR.isin(ARs)].assign(value=lambda d: d.value),
        x='t_mean',
        y='value',
        hue='variable',
        estimator='median',
        errorbar='ci',
        err_style='band',
        marker='^',
        linewidth=2,
        linestyle='dashed',
        markersize=10,
        alpha=0.5,
        ax=ax2
    )

    ax2.set_ylim(-10, 10)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_ylabel(r'Acummulated $\gamma$ [deg]', fontsize=12)
    ax2.legend().remove()

    # --- Títulos específicos ---
    if i == 0:
        ax1.set_title(f"All ARs ({len(ARs)})", fontsize=14)
    elif i == 1:
        ax1.set_title(f"High flux ARs ({len(ARs)})", fontsize=14)
    else:
        ax1.set_title(f"Low flux ARs ({len(ARs)})", fontsize=14)

# Etiqueta común para el eje x
axes[-1].set_xlabel(r'$t_{norm}$', fontsize=12)

plt.tight_layout()
plt.show()

# --- Nueva celda ---





# --- Nueva celda ---



# --- Nueva celda ---

DFx2 = DFx.assign(t_mean=lambda x: round(10 * x.t_norm) / 10)

# --- tomar alpha y alphab al máximo (t_mean=1) ---
DFmax_alpha = (
    DFx2[DFx2.t_mean == 1]
    .groupby("AR")[["alpha", "alphab"]]
    .mean()
    .reset_index()
    .rename(columns={"alpha": "alpha_max", "alphab": "alphab_max"})
)

DFx2 = DFx2.merge(DFmax_alpha, on="AR", how="left")


DFx2 = DFx2.assign(fraca=lambda x: np.sign(x.alpha)*x.rot)
DFx2 = DFx2.assign(fracb=lambda x: np.sign(x.alphab)*x.rotb)

# --- merge con DFx2 ---
DFx2 = DFx2.assign(rot_cum=lambda x: np.sign(x.alpha)*x.rot_cum * 1.5 * x.stp)
DFx2 = DFx2.assign(rotb_cum=lambda x: np.sign(x.alphab)*x.rotb_cum * 1.5 * x.stp)

DFmax = DFx2[DFx2.t_mean == 1].groupby('AR').mean().reset_index()
ARup = DFmax[DFmax.flux >= np.median(DFmax.flux)].AR.unique()
ARlow = DFmax[DFmax.flux < np.median(DFmax.flux)].AR.unique()

# --- 3 filas, 1 columna ---
fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True, sharey=True)

for i, ARs in enumerate([DFmax.AR.unique(), ARup, ARlow]):
    ax1 = axes[i]

    # --- rot y rotb ---
    df1 = pd.melt(
        DFx2,
        id_vars=['AR', 't_mean'],
        value_vars=['fraca', 'fracb'],
        var_name='variable',
        value_name='value'
    )

    sns.lineplot(
        data=df1[df1.AR.isin(ARs)].assign(value=lambda d: d.value),
        x='t_mean',
        y='value',
        estimator='median',
        hue='variable',
        errorbar='ci',
        err_style='bars',
        marker='o',
        linewidth=3,
        markersize=10,
        alpha=0.5,
        ax=ax1
    )

    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.2, 0.2)
    ax1.axhline(0, color='black', linestyle='dashed')
    ax1.set_ylabel(r'$\gamma $ [deg/hour]', fontsize=12)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.legend(loc='upper right', fontsize=10)

    # --- Segundo eje: rot_cum y rotb_cum ---
    ax2 = ax1.twinx()

    df2 = pd.melt(
        DFx2,
        id_vars=['AR', 't_mean'],
        value_vars=['rot_cum', 'rotb_cum'],
        var_name='variable',
        value_name='value'
    )

    sns.lineplot(
        data=df2[df2.AR.isin(ARs)].assign(value=lambda d: d.value),
        x='t_mean',
        y='value',
        hue='variable',
        estimator='median',
        errorbar='ci',
        err_style='band',
        marker='^',
        linewidth=2,
        linestyle='dashed',
        markersize=10,
        alpha=0.5,
        ax=ax2
    )

    ax2.set_ylim(-10, 10)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_ylabel(r'Acummulated $\gamma$ [deg]', fontsize=12)
    ax2.legend().remove()

    # --- Títulos específicos ---
    if i == 0:
        ax1.set_title(f"All ARs ({len(ARs)})", fontsize=14)
    elif i == 1:
        ax1.set_title(f"High flux ARs ({len(ARs)})", fontsize=14)
    else:
        ax1.set_title(f"Low flux ARs ({len(ARs)})", fontsize=14)

# Etiqueta común para el eje x
axes[-1].set_xlabel(r'$t_{norm}$', fontsize=12)

plt.tight_layout()
plt.show()


# --- Nueva celda ---

sns.scatterplot(data=np.abs(DFx2).groupby('AR').max(),x='frac',y='Nt')
#sns.scatterplot(data=DFx2.groupby('AR').mean(),x='rotb',y='Nt')

#plt.xlim(-1,1)
#plt.ylim(-1,1)

# --- Nueva celda ---

plt.figure(figsize=(20,5))
sns.barplot(data=DFx2[DFx2.frac > 20].groupby(['AR']).mean().reset_index(),x='AR',y='frac')

plt.xticks(rotation=45)
plt.show()


# --- Nueva celda ---

DFx2=DFx.assign(frac=lambda x: np.abs(x.alpha-x.alphab))
DFx2=DFx2.assign(t_mean=lambda x: round(10*x.t_norm)/10)
#

# --- Nueva celda ---

for name in DFx2[DFx2.frac > 30].AR.unique():

  sns.lineplot(data=pd.melt(DFx2[DFx2.AR==name],id_vars=['t_mean','AR'],value_vars=['alpha','alphab'],var_name='variable'),
             x='t_mean',y='value',hue='variable')
  plt.title(name)
  plt.show()

# --- Nueva celda ---

sns.scatterplot(data=np.abs(DFx2).groupby('AR').max(),x='rotb',y='flux')
#sns.scatterplot(data=DFx2.groupby('AR').mean(),x='rotb',y='Nt')

plt.xlim(0,5)
#plt.ylim(-1,1)

# --- Nueva celda ---

DFx2=DFx.assign(frac=lambda x: x.alpha-x.alphab)
DFx2=DFx2.assign(t_mean=lambda x: round(10*x.t_norm)/10)
#
#
#  Jointplot de seaborn con histograma extendido en marg_y
g = sns.jointplot(
    data=DFx2,
    x='t_norm',
    y='frac',
    kind='scatter',
    alpha=0.2,
    marginal_kws={'bins': 30, 'fill': True},ratio=2,height=6,space=0.3
)

DFx2=DFx2.groupby(['AR','t_mean']).mean().reset_index()
sns.lineplot(
data=DFx2,
x='t_mean',
y='frac',
ax=g.ax_joint,
estimator='mean',
errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='band',   # o 'bars' para barras verticales
marker='o'          # opcional para marcar puntos medios
)

g.ax_marg_x.set_visible(False)
g.ax_marg_y.clear()
hist=sns.histplot(DFx2[DFx2['t_mean'].isin([0.1,0.5,0.9])],y='frac',
             hue='t_mean',palette=['tab:blue','tab:orange','tab:green'],
             alpha=0.7,ax=g.ax_marg_y)

#g.ax_marg_y.legend()
g.ax_marg_y.legend(title=r'$t_{norm}$',labels=['0.9','0.5','0.1'],loc='upper right')
handles,labels=hist.get_legend_handles_labels()
#hist.legend(handles=handles,labels=labels,title=r'$t_{norm}$')
handles,labels=hist.get_legend_handles_labels()
g.ax_marg_y.set_ylabel('')
g.ax_marg_y.set_xlabel('ARs')
g.ax_joint.plot([0,1],[50,50],color='black')
g.ax_joint.plot([1,1],[-50,50],color='black')
g.ax_marg_y.plot([0,50],[-50,-50],color='black')
g.ax_marg_y.plot([0,50],[50,50],color='black')
g.ax_marg_y.plot([50,50],[-50,50],color='black')


#g.ax_marg_y.legend(handles,['0.1','0.5','0.9'],title=r'$t_{norm}$')
g.ax_marg_y.set_ylim(-50, 50)
g.ax_joint.set_xlim(0, 1)
g.ax_joint.set_ylim(-50, 50)

stds=DFx2[DFx2['t_mean'].isin([0.1,0.5,0.9])].groupby('t_mean').std()['frac'].values

g.ax_marg_y.text(20,-30,rf'$\sigma=${stds[0]:.0f}$^\circ$',color='tab:blue')
g.ax_marg_y.text(20,-25,rf'$\sigma=${stds[1]:.0f}$^\circ$',color='tab:orange')
g.ax_marg_y.text(20,-20,rf'$\sigma=${stds[2]:.0f}$^\circ$',color='tab:green')

g.ax_joint.axhline(0,color='black')
g.set_axis_labels(r'$t_{norm}$', r'$\alpha_{mod}-\alpha_{bar}$  [deg]')

#plt.savefig('./plots/errorhist.pdf',dpi=300)


# --- Nueva celda ---

set(DFx2.t_mean.values)

# --- Nueva celda ---

DFx2=DFx.assign(frac=lambda x: x.alpha-x.alphab)
DFx2=DFx2.assign(t_mean=lambda x: round(10*x.t_norm)/10)
DFx2=DFx2.groupby(['AR','t_mean']).mean().reset_index()
DFx2=DFx2[DFx2.t_mean<=1]
#
#
#  Jointplot de seaborn con histograma extendido en marg_y
g = sns.jointplot(
    data=DFx2,
    x='alphab',
    y='alpha',
    kind='scatter',
    alpha=0.7,
    marginal_kws={'bins': 30,'binrange':(-80,80), 'fill': True},
    ratio=2,height=6,space=0.3
)

g.set_axis_labels(r'$\alpha_\mathrm{bar}$ [deg]',r'$\alpha_\mathrm{mod}$ [deg]', fontsize=12)

g.ax_marg_x.set_visible(False)
#g.ax_marg_y.set_visible(False)
g.ax_marg_x.clear()
g.ax_marg_y.clear()
g.ax_marg_x.set_xlabel('X axis label')
g.ax_marg_y.set_ylabel('Y axis label')
g.ax_marg_x.tick_params(axis='x', labelsize=12)
g.ax_marg_y.tick_params(axis='y', labelsize=12)

hist=sns.histplot(data=pd.melt(DFx2,id_vars=['t_norm','lat','AR','fn','mag','flux'],value_vars=['alpha','alphab']),y='value',
                  hue='variable',
             palette=['tab:blue','tab:orange'],
             alpha=0.5,ax=g.ax_marg_y,binrange=(-80,80),bins=40)

# Linear fit
slope, intercept, r_value, p_value, std_err = stats.linregress(DFx2['alphab'], DFx2['alpha'])

# Add linear fit line
x_fit = np.linspace(DFx2['alphab'].min(), DFx2['alphab'].max(), 100)
y_fit = slope * x_fit + intercept
g.ax_joint.plot(x_fit, y_fit, color='red', label=f'Fit: y={slope:.2f}x+{intercept:.2f}')

# Annotate with correlation coefficient and slope
g.ax_joint.text(
    0.65, 0.35,
    f'$r$ = {r_value:.2f}\nSlope = {slope:.2f}',
    transform=g.ax_joint.transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle="round", alpha=0.2, color='red')
)

#g.ax_marg_y.legend()
g.ax_marg_y.legend(title='',labels=[r'$\alpha_\mathrm{bar}$',r'$\alpha_\mathrm{mod}$'],loc='upper right')
handles,labels=hist.get_legend_handles_labels()
#hist.legend(handles=handles,labels=labels,title=r'$t_{norm}$')
handles,labels=hist.get_legend_handles_labels()
g.ax_marg_y.set_ylabel('')
#
g.ax_marg_y.tick_params(axis='y', labelsize=10)  # Set y-ticklabel fontsize
g.ax_marg_y.tick_params(axis='x', labelsize=10)  # Set x-ticklabel fontsize (if needed)

g.ax_marg_y.axhline(0,color='black')

g.ax_marg_y.axhline(DFx2.alpha.mean(),color='tab:blue')
g.ax_marg_y.axhline(DFx2.alphab.mean(),color='tab:orange')

g.ax_joint.axvline(0,color='black',linestyle='dashed')
g.ax_joint.axhline(0,color='black',linestyle='dashed')
g.ax_joint.set_ylim(-80, 80)
g.ax_joint.set_xlim(-80, 80)

g.ax_marg_y.text(
    0.45, 0.15,
    f'$\sigma_\mathrm{{mod}}$ = {DFx2.alpha.std():.0f}\n$\sigma_\mathrm{{bar}}$ = {DFx2.alphab.std():.0f}',
    transform=g.ax_marg_y.transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle="round", alpha=0.0, color='red')
)

g.ax_marg_y.set_xlim(0,200)

g.ax_joint.plot([-80,80],[80,80],color='black')
g.ax_joint.plot([80,80],[-80,80],color='black')
g.ax_marg_y.plot([0,200],[-80,-80],color='black')
g.ax_marg_y.plot([0,200],[80,80],color='black')
g.ax_marg_y.plot([200,200],[-80,80],color='black')


aa=DFx2.alpha.values
ab=DFx2.alphab.values



print(round(100*sum((aa>0)*(ab>0))/len(aa)))
print(round(100*sum((aa<0)*(ab<0))/len(aa)))
print(round(100*sum((aa>0)*(ab<0))/len(aa)))
print(round(100*sum((aa<0)*(ab>0))/len(aa)))

g.ax_joint.text(0.7,0.9,f'~{round(100*sum((aa>0)*(ab>0))/len(aa))}%',transform=g.ax_joint.transAxes,
                fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle="round", alpha=0.5, color='tab:blue'))
g.ax_joint.text(0.15,0.9,f'~{round(100*sum((aa>0)*(ab<0))/len(aa))}%',transform=g.ax_joint.transAxes,
                fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle="round", alpha=0.5, color='tab:blue'))
g.ax_joint.text(0.7,0.2,f'~{round(100*sum((aa<0)*(ab>0))/len(aa))}%',transform=g.ax_joint.transAxes,
                fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle="round", alpha=0.5, color='tab:blue'))
g.ax_joint.text(0.15,0.2,f'~{round(100*sum((aa<0)*(ab<0))/len(aa))}%',transform=g.ax_joint.transAxes,
                fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle="round", alpha=0.5, color='tab:blue'))
plt.savefig('./plots/alpha-alpha.pdf',dpi=300)


# --- Nueva celda ---

aa=DFx2.alpha.values
ab=DFx2.alphab.values



print(round(100*sum((aa>0)*(ab>0))/len(aa)))
print(round(100*sum((aa<0)*(ab<0))/len(aa)))
print(round(100*sum((aa>0)*(ab<0))/len(aa)))
print(round(100*sum((aa<0)*(ab>0))/len(aa)))



print(len(aa))


# --- Nueva celda ---

DFx2=DFx.assign(frac=lambda x: np.abs(x.alpha-x.alphab))
DFx2=DFx2.assign(t_mean=lambda x: round(10*x.t_norm)/10)
#
#
#  Jointplot de seaborn con histograma extendido en marg_y
g = sns.jointplot(
    data=DFx2,
    x='t_norm',
    y='frac',
    kind='scatter',
    alpha=0.2,
    marginal_kws={'bins': 30, 'fill': True},ratio=2,height=6,space=0.3
)

DFx2=DFx2.groupby(['AR','t_mean']).mean().reset_index()
sns.lineplot(
data=DFx2,
x='t_mean',
y='frac',
ax=g.ax_joint,
estimator='mean',
errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='band',   # o 'bars' para barras verticales
marker='o'          # opcional para marcar puntos medios
)

g.ax_marg_x.set_visible(False)
g.ax_marg_y.clear()
hist=sns.histplot(DFx2[DFx2['t_mean'].isin([0.1,0.5,0.9])],y='frac',
             hue='t_mean',palette=['tab:blue','tab:orange','tab:green'],
             alpha=0.7,ax=g.ax_marg_y)

#g.ax_marg_y.legend()
g.ax_marg_y.legend(title=r'$t_{norm}$',labels=['0.9','0.5','0.1'],loc='upper right')
handles,labels=hist.get_legend_handles_labels()
#hist.legend(handles=handles,labels=labels,title=r'$t_{norm}$')
handles,labels=hist.get_legend_handles_labels()
g.ax_marg_y.set_ylabel('')
g.ax_marg_y.set_xlabel('ARs')
g.ax_joint.plot([0,1],[50,50],color='black')
g.ax_joint.plot([1,1],[0,50],color='black')
g.ax_marg_y.plot([0,50],[0,0],color='black')
g.ax_marg_y.plot([0,50],[50,50],color='black')
g.ax_marg_y.plot([50,50],[0,50],color='black')


#g.ax_marg_y.legend(handles,['0.1','0.5','0.9'],title=r'$t_{norm}$')
g.ax_marg_y.set_ylim(0, 50)
g.ax_joint.set_xlim(0, 1)
g.ax_joint.set_ylim(0, 50)

stds=DFx2[DFx2['t_mean'].isin([0.1,0.5,0.9])].groupby('t_mean').std()['frac'].values
means=DFx2[DFx2['t_mean'].isin([0.1,0.5,0.9])].groupby('t_mean').mean()['frac'].values


g.ax_marg_y.text(10,24,rf'$\mu=${means[0]:.0f}$^\circ$, $\sigma=${stds[0]:.0f}$^\circ$',color='tab:blue')
g.ax_marg_y.text(10,27,rf'$\mu=${means[1]:.0f}$^\circ$, $\sigma=${stds[1]:.0f}$^\circ$',color='tab:orange')
g.ax_marg_y.text(10,30,rf'$\mu=${means[2]:.0f}$^\circ$, $\sigma=${stds[2]:.0f}$^\circ$',color='tab:green')

g.ax_joint.axhline(0,color='black')
g.set_axis_labels(r'$t_{norm}$', r'|$\alpha_{mod}-\alpha_{bar}$|  [deg]')

plt.savefig('./plots/errorhistabs.pdf',dpi=300)


# --- Nueva celda ---

sns.scatterplot(data=DFx2,x='alphab',y='alpha',alpha=0.7,color='tab:blue')

# --- Nueva celda ---



# --- Nueva celda ---

DFx=DFx.assign(rot= lambda x: np.gradient(x.alpha,x.mag)/1.5)
DFx=DFx.assign(rotb= lambda x: np.gradient(x.alphab,x.mag)/1.5)

DFx2=DFx.assign(frac=lambda x: x.rot-x.rotb)
DFx2=DFx2.assign(t_mean=lambda x: round(10*x.t_norm)/10)
#
#
#  Jointplot de seaborn con histograma extendido en marg_y
g = sns.jointplot(
    data=DFx2,
    x='t_norm',
    y='rot',
    kind='scatter',
    alpha=0.2,
    marginal_kws={'bins': 30, 'fill': True},ratio=2,height=6,space=0.7
)

sns.scatterplot(data=DFx2,x='t_norm',y='rotb',alpha=0.2,color='tab:orange')

DFx2=DFx2.groupby(['AR','t_mean']).mean().reset_index()
sns.lineplot(
data=DFx2,
x='t_mean',
y='rot',
ax=g.ax_joint,
estimator='mean',
errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='band',   # o 'bars' para barras verticales
marker='o',          # opcional para marcar puntos medios
label=r'$\gamma_\mathrm{mod}$')
sns.lineplot(
data=DFx2,
x='t_mean',
y='rotb',
ax=g.ax_joint,
estimator='mean',
errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='band',   # o 'bars' para barras verticales
marker='o',          # opcional para marcar puntos medios
label=r'$\gamma_\mathrm{bar}$'
)

g.ax_marg_x.set_visible(False)
g.ax_marg_y.clear()
'''
hist=sns.histplot(DFx2[DFx2['t_mean']<=0.5],y='rot',
             color='tab:blue',
             alpha=0.7,ax=g.ax_marg_y,bins=30,binrange=(-5,5))
hist=sns.histplot(DFx2[DFx2['t_mean']<=0.5],y='rotb',
             color='tab:orange',
             alpha=0.7,ax=g.ax_marg_y,bins=30,binrange=(-5,5))
'''


hist=sns.histplot(DFx2[DFx2['t_mean']<=1],y='frac',
             color='tab:blue',
             alpha=0.7,ax=g.ax_marg_y,bins=30,binrange=(-2,2))

#g.legend(title='',loc='upper right')
#g.ax_marg_y.legend()
#g.ax_marg_y.legend(title=r'$t_{norm}$',labels=['0.1','0.5','0.9'],loc='upper right')
handles,labels=hist.get_legend_handles_labels()
g.ax_marg_y.set_ylabel(r'$\gamma_\mathrm{mod}-\gamma_\mathrm{bar}$ [deg/hour]')
g.ax_marg_y.set_xlabel('Count')
g.ax_joint.plot([0,1],[2,2],color='black')
g.ax_joint.plot([1,1],[-2,2],color='black')
g.ax_marg_y.plot([0,300],[-2,-2],color='black')
g.ax_marg_y.plot([0,300],[2,2],color='black')
g.ax_marg_y.plot([300,300],[-2,2],color='black')


#g.ax_marg_y.legend(handles,['0.1','0.5','0.9'],title=r'$t_{norm}$')
g.ax_marg_y.set_ylim(-2, 2)
g.ax_marg_y.set_xlim(0, 300)
g.ax_joint.set_xlim(0, 1)
g.ax_joint.set_ylim(-2, 2)
g.ax_joint.tick_params(axis='both', labelsize=12)
g.ax_marg_y.tick_params(axis='both', labelsize=12)

stds=DFx2[DFx2['t_mean'].isin([0.1])].groupby('t_mean').std()[['rot','rotb']].values

#g.ax_marg_y.text(20,-2,rf'$\sigma=${stds[0][0]:.1f}$^\circ$',color='tab:blue')
#g.ax_marg_y.text(20,-2.5,rf'$\sigma=${stds[0][1]:.1f}$^\circ$',color='tab:orange')

#g.ax_marg_y.legend(handles=handles,title='$t_\mathrm{norm}$',loc='upper right')
g.ax_joint.axhline(0,color='black')
g.set_axis_labels(r'$t_{norm}$', r'$\gamma$  [deg/hour]',fontsize=12)

plt.savefig('./plots/rotations.pdf',dpi=300)


# --- Nueva celda ---

DFx=DFx.assign(rot= lambda x: np.gradient(x.alpha,x.mag)/1.5)
DFx=DFx.assign(rotb= lambda x: np.gradient(x.alphab,x.mag)/1.5)

DFx2=DFx.assign(frac=lambda x: x.rot-x.rotb)
DFx2=DFx2.assign(t_mean=lambda x: round(10*x.t_norm)/10)
#
#
#  Jointplot de seaborn con histograma extendido en marg_y
g = sns.jointplot(
    data=DFx2,
    x='t_norm',
    y='frac',
    kind='scatter',
    alpha=0.2,
    marginal_kws={'bins': 30, 'fill': True},ratio=2,height=6,space=0.3
)

DFx2=DFx2.groupby(['AR','t_mean']).mean().reset_index()
sns.lineplot(
data=DFx2,
x='t_mean',
y='frac',
ax=g.ax_joint,
estimator='mean',
errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='band',   # o 'bars' para barras verticales
marker='o'          # opcional para marcar puntos medios
)

g.ax_marg_x.set_visible(False)
g.ax_marg_y.clear()
hist=sns.histplot(DFx2[DFx2['t_mean'].isin([0.1,0.5,0.9])],y='frac',
             hue='t_mean',palette=['tab:blue','tab:orange','tab:green'],
             alpha=0.7,ax=g.ax_marg_y,bins=30,binrange=(-5,5))

#g.ax_marg_y.legend()
g.ax_marg_y.legend(title=r'$t_{norm}$',labels=['0.1','0.5','0.9'],loc='upper right')
handles,labels=hist.get_legend_handles_labels()
g.ax_marg_y.set_ylabel('')
g.ax_marg_y.set_xlabel('ARs')
g.ax_joint.plot([0,1],[5,5],color='black')
g.ax_joint.plot([1,1],[-5,5],color='black')
g.ax_marg_y.plot([0,50],[-5,-5],color='black')
g.ax_marg_y.plot([0,50],[5,5],color='black')
#g.ax_marg_y.plot([50,50],[-5,5],color='black')


#g.ax_marg_y.legend(handles,['0.1','0.5','0.9'],title=r'$t_{norm}$')
g.ax_marg_y.set_ylim(-5, 5)
g.ax_joint.set_xlim(0, 1)
g.ax_joint.set_ylim(-5, 5)

stds=DFx2[DFx2['t_mean'].isin([0.1,0.5,0.9])].groupby('t_mean').std()['frac'].values

g.ax_marg_y.text(20,-2,rf'$\sigma=${stds[0]:.1f}$^\circ$',color='tab:blue')
g.ax_marg_y.text(20,-2.5,rf'$\sigma=${stds[1]:.1f}$^\circ$',color='tab:orange')
g.ax_marg_y.text(20,-3,rf'$\sigma=${stds[2]:.1f}$^\circ$',color='tab:green')

g.ax_joint.axhline(0,color='black')
g.set_axis_labels(r'$t_{norm}$', r'$\gamma_{mod} - \gamma_{bar}$  [deg/hour]')

plt.savefig('./plots/rotdif.pdf',dpi=300)


# --- Nueva celda ---

DFx=DFx.assign(rot= lambda x: np.gradient(x.alpha,x.mag)/1.5)
DFx=DFx.assign(rotb= lambda x: np.gradient(x.alphab,x.mag)/1.5)

DFx2=DFx.assign(frac=lambda x: np.abs(x.rot)-np.abs(x.rotb))
DFx2=DFx2.assign(t_mean=lambda x: round(10*x.t_norm)/10)
#
#
#  Jointplot de seaborn con histograma extendido en marg_y
g = sns.jointplot(
    data=DFx2,
    x='t_norm',
    y='frac',
    kind='scatter',
    alpha=0.2,
    marginal_kws={'bins': 30, 'fill': True},ratio=2,height=6,space=0.3
)

DFx2=DFx2.groupby(['AR','t_mean']).mean().reset_index()
sns.lineplot(
data=DFx2,
x='t_mean',
y='frac',
ax=g.ax_joint,
estimator='mean',
errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='band',   # o 'bars' para barras verticales
marker='o'          # opcional para marcar puntos medios
)

g.ax_marg_x.set_visible(False)
g.ax_marg_y.clear()
hist=sns.histplot(DFx2[DFx2['t_mean'].isin([0.1,0.5,0.9])],y='frac',
             hue='t_mean',palette=['tab:blue','tab:orange','tab:green'],
             alpha=0.7,ax=g.ax_marg_y,bins=30,binrange=(-5,5))

#g.ax_marg_y.legend()
g.ax_marg_y.legend(title=r'$t_{norm}$',labels=['0.1','0.5','0.9'],loc='upper right')
handles,labels=hist.get_legend_handles_labels()
g.ax_marg_y.set_ylabel('')
g.ax_marg_y.set_xlabel('ARs')
g.ax_joint.plot([0,1],[5,5],color='black')
g.ax_joint.plot([1,1],[-5,5],color='black')
g.ax_marg_y.plot([0,50],[-5,-5],color='black')
g.ax_marg_y.plot([0,50],[5,5],color='black')
#g.ax_marg_y.plot([50,50],[-5,5],color='black')


#g.ax_marg_y.legend(handles,['0.1','0.5','0.9'],title=r'$t_{norm}$')
g.ax_marg_y.set_ylim(-5, 5)
g.ax_joint.set_xlim(0, 1)
g.ax_joint.set_ylim(-5, 5)

stds=DFx2[DFx2['t_mean'].isin([0.1,0.5,0.9])].groupby('t_mean').std()['frac'].values

g.ax_marg_y.text(20,-2,rf'$\sigma=${stds[0]:.1f}$^\circ$',color='tab:blue')
g.ax_marg_y.text(20,-2.5,rf'$\sigma=${stds[1]:.1f}$^\circ$',color='tab:orange')
g.ax_marg_y.text(20,-3,rf'$\sigma=${stds[2]:.1f}$^\circ$',color='tab:green')

g.ax_joint.axhline(0,color='black')
g.set_axis_labels(r'$t_{norm}$', r'$|\gamma_{mod}| - |\gamma_{bar}|$  [deg/hour]')

plt.savefig('./plots/rotdifabs.pdf',dpi=300)


# --- Nueva celda ---

stds[0][0]

# --- Nueva celda ---


DFx2=DFx.assign(frac=lambda x: 2*np.abs(x.alpha-x.alphab)/np.abs(x.alpha+x.alphab))
DFx2=DFx2.assign(t_mean=lambda x: round(10*x.t_norm)/10)


g=sns.scatterplot(data=DFx2,x='t_norm',y='frac',alpha=0.2)

DFx2=DFx2.groupby(['AR','t_mean']).mean().reset_index()
g=sns.lineplot(
data=DFx2,
x='t_mean',
y='frac',
estimator='mean',
errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
err_style='band',   # o 'bars' para barras verticales
marker='o'          # opcional para marcar puntos medios
)
#   sns.lineplot(data=DFx2.groupby(['t_norm','variable']).median().reset_index(),x='t_norm',y='value',hue='variable')
plt.xlim(0,1)
plt.ylim(0,10)
'''
plt.xlim(0,1)
plt.ylim(-200,200)

plt.ylabel(r'$\alpha / \sin$(lat) [deg]')
plt.xlabel(r'$t_{norm}$')

plt.axhline(0,linestyle='dashed',color='gray')

plt.title(fr'Latitude range: {r[0]}$^\circ -$  {r[1]}$^\circ$')

handles, labels = g.get_legend_handles_labels()

#   plt.legend(title='',handles=handles,labels=labels[:2])

plt.legend(title='',handles=handles,labels=['Bayes','Barycenters'])
plt.show()

'''

# --- Nueva celda ---

DFx1=pd.melt(DFx,id_vars=['t_norm','lat','AR','fn','mag','flux','Nt'],value_vars=['alpha','alphab'])

# --- Nueva celda ---

#DFx1=pd.concat([DFx1,DFx0b]).reset_index()

# --- Nueva celda ---

DFx1=DFx1.assign(t_mean=lambda x: round(10*x.t_norm)/10)

# --- Nueva celda ---

DFx1['sin_lat'] = np.sin(np.radians(np.abs(DFx1['lat'])))
DFx1['tilt_sinlat'] = DFx1['value'] / DFx1['sin_lat']

# --- Nueva celda ---

DFx1=DFx1.assign(tilt_norm=lambda x: x.value/np.sin(np.pi*np.abs(x.lat)/180))

# --- Nueva celda ---

DFx1 = DFx1[np.abs(DFx1['sin_lat']) > 0.05].copy()

# --- Nueva celda ---

DFx

# --- Nueva celda ---

list(set(DFx1.t_mean))[:10]

# --- Nueva celda ---

ARout=[]

for tt in list(set(DFx1.t_mean))[:10]:

    df=DFx1[(DFx1.variable=='alpha') & (DFx1.t_mean==tt)]
    dfar=df.groupby(['AR','variable']).mean().reset_index()
    # Median and MAD
    median_tilt_sinlat = dfar['tilt_sinlat'].median()
    mad_tilt_sinlat = np.median(np.abs(dfar['tilt_sinlat'] - median_tilt_sinlat))

    # Define threshold (e.g., 3 times MAD)
    threshold = 2
    modified_z_scores = 0.6745 * (dfar['tilt_sinlat'] - median_tilt_sinlat) / mad_tilt_sinlat

    # Filter out outliers
    dfar_clean = dfar[np.abs(modified_z_scores) >= threshold].copy()

    for names in dfar_clean.AR.values:
        ARout.append(names)

# --- Nueva celda ---

ARout=list(set(ARout))

# --- Nueva celda ---

len(ARout)

# --- Nueva celda ---

DFmax

# --- Nueva celda ---

DFmax=DFx1.groupby(['AR','variable']).max().reset_index()
DFmax=DFmax.assign(heli=lambda x: np.abs(x.Nt)*x.flux**2)
DFmax['B_bin']=pd.qcut(DFmax['flux'],q=3,labels=['low','medium','high'])
DFmax['H_bin']=pd.qcut(DFmax['heli'],q=3,labels=['low','medium','high'])

#DFmax['B_bin'] = pd.cut(DFmax['flux'], bins=3, labels=['low', 'medium', 'high']) # for equal lenght bins
DFx1 = DFx1.merge(DFmax[['AR', 'variable', 'heli','B_bin','H_bin']], on=['AR', 'variable'], how='left')

# --- Nueva celda ---

sns.boxplot(DFmax,x='B_bin',y='flux')

# --- Nueva celda ---

DFx1

# --- Nueva celda ---

DFall = []

for r in [(0,39)]:
    DFx2 = DFx1[(np.abs(DFx1.lat) >= r[0]) & (np.abs(DFx1.lat) < r[1])]
    #DFx2 = DFx2[~DFx2['AR'].isin(ARout)].copy()

    # agrupás por AR en cada bin temporal
    DF_AR = (
        DFx2.groupby(['AR','variable','t_mean'])
        .agg(value_mean=('value','mean'),
             value_std=('value','std'),
             count=('value','count'))
        .reset_index()
    )

    # ahora promedio ponderado entre ARs por bin
    out = []
    for (var,t), g in DF_AR.groupby(['variable','t_mean']):
        # pesos = 1/sigma^2 (si sigma=NaN porque solo hay 1 punto, asignás sigma grande)
        sigma = g['value_std'].fillna(g['value_mean'].std() if len(g)>1 else 999)
        weights = 1 / (sigma**2)
        mean = np.average(g['value_mean'], weights=weights)
        err = np.sqrt(1/weights.sum())
        out.append([var, t, mean, err])

    DF_plot = pd.DataFrame(out, columns=['variable','t_mean','tilt','err'])
    DF_plot['lat_range'] = f"{r[0]}-{r[1]}"

    DFall.append(DF_plot)

DFall = pd.concat(DFall)

# --- Nueva celda ---

DFx1[DFx1.AR==8193]

# --- Nueva celda ---


for r in DFall.lat_range.unique():
    g = sns.scatterplot(
        data=DFall[DFall.lat_range==r],
        x='t_mean', y='tilt',
        hue='variable', style='variable',
        marker='o'
    )

    for var,sub in DFall[DFall.lat_range==r].groupby('variable'):
        plt.errorbar(sub.t_mean, sub.tilt, yerr=sub.err,
                     fmt='o-', capsize=3, label=var)

    plt.axhline(0, linestyle='dashed', color='gray')
    plt.xlim(0,1)
    plt.ylim(-40,40)
    plt.ylabel(r'$\alpha$ [deg]')
    plt.xlabel(r'$t_{norm}$')
    plt.title(f'Latitude range: {r}°')

    plt.legend()
    plt.show()


# --- Nueva celda ---

for r in [(0,39),(15,18),(18,21)]:
    DFx2=DFx1[(np.abs(DFx1.lat) >= r[0]) & (np.abs(DFx1.lat) < r[1])]

    #DFx2 = DFx2[~DFx2['AR'].isin(ARout)].copy()
    print(f'range: {r} / ARs: {len(set(DFx2.AR))}')

    g=sns.scatterplot(data=DFx2,x='t_norm',y='value',hue='variable',alpha=0.2)
    DFx2=DFx2.groupby(['AR','variable','t_mean']).mean(numeric_only=True).reset_index()
    g=sns.lineplot(
        data=DFx2,
        x='t_mean',
        y='value',
        hue='variable',
        estimator='median',
        errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
        err_style='band',   # o 'bars' para barras verticales
        marker='o'          # opcional para marcar puntos medios
    )
 #   sns.lineplot(data=DFx2.groupby(['t_norm','variable']).median().reset_index(),x='t_norm',y='value',hue='variable')

    plt.xlim(0,1)
    plt.ylim(-40,40)

    plt.ylabel(r'$\alpha $[deg]')
    plt.xlabel(r'$t_{norm}$')

    plt.axhline(0,linestyle='dashed',color='gray')

    plt.title(fr'Latitude range: {r[0]}$^\circ -$  {r[1]}$^\circ$')

    handles, labels = g.get_legend_handles_labels()

 #   plt.legend(title='',handles=handles,labels=labels[:2])

    plt.legend(title='',handles=handles[2:4],labels=['Bayes','Barycenters'])
    plt.show()

# --- Nueva celda ---

for r in ['low','medium','high']:
    DFx2=DFx1[DFx1.B_bin==r]

   # DFx2 = DFx2[~DFx2['AR'].isin(ARout)].copy()
    print(f'B range: {r} / ARs: {len(set(DFx2.AR))}')

    g=sns.scatterplot(data=DFx2,x='t_norm',y='tilt_sinlat',hue='variable',alpha=0.2)
    DFx2=DFx2.groupby(['AR','variable','t_mean']).mean(numeric_only=True).reset_index()
    g=sns.lineplot(
        data=DFx2,
        x='t_mean',
        y='tilt_sinlat',
        hue='variable',
        estimator='median',
        errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
        err_style='band',   # o 'bars' para barras verticales
        marker='o'          # opcional para marcar puntos medios
    )
 #   sns.lineplot(data=DFx2.groupby(['t_norm','variable']).median().reset_index(),x='t_norm',y='value',hue='variable')

    plt.xlim(0,1)
    plt.ylim(-200,200)

    plt.ylabel(r'$\alpha / \sin(lat)$ [deg]')
    plt.xlabel(r'$t_{norm}$')

    plt.axhline(0,linestyle='dashed',color='gray')
    minf=DFmax[DFmax.B_bin==r]['flux'].min()
    maxf=DFmax[DFmax.B_bin==r]['flux'].max()
    plt.title(f'B range: {minf:.2e} -- {maxf:.2e}')

    handles, labels = g.get_legend_handles_labels()

 #   plt.legend(title='',handles=handles,labels=labels[:2])

    plt.legend(title='',handles=handles,labels=['Bayes','Barycenters'])
    plt.show()


# --- Nueva celda ---

DFx2.groupby(['AR','variable','t_mean']).mean(numeric_only=True)

# --- Nueva celda ---

for r in ['low','medium','high']:
    DFx2=DFx1[DFx1.H_bin==r]

   # DFx2 = DFx2[~DFx2['AR'].isin(ARout)].copy()
    print(f'B range: {r} / ARs: {len(set(DFx2.AR))}')

    g=sns.scatterplot(data=DFx2,x='t_norm',y='tilt_sinlat',hue='variable',alpha=0.2)
    DFx2=DFx2.groupby(['AR','variable','t_mean']).median(numeric_only=True).reset_index()
    g=sns.lineplot(
        data=DFx2,
        x='t_mean',
        y='tilt_sinlat',
        hue='variable',
        estimator='median',
        errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
        err_style='band',   # o 'bars' para barras verticales
        marker='o'          # opcional para marcar puntos medios
    )
 #   sns.lineplot(data=DFx2.groupby(['t_norm','variable']).median().reset_index(),x='t_norm',y='value',hue='variable')

    plt.xlim(0,1)
    plt.ylim(-200,200)

    plt.ylabel(r'$\alpha / \sin(lat)$ [deg]')
    plt.xlabel(r'$t_{norm}$')

    plt.axhline(0,linestyle='dashed',color='gray')
    minf=DFmax[DFmax.H_bin==r]['heli'].min()
    maxf=DFmax[DFmax.H_bin==r]['heli'].max()
    plt.title(f'Twist range: {minf:.2e} -- {maxf:.2e} Mx²')

    handles, labels = g.get_legend_handles_labels()

 #   plt.legend(title='',handles=handles,labels=labels[:2])

    plt.legend(title='',handles=handles,labels=['Bayes','Barycenters'])
    plt.show()


# --- Nueva celda ---

for r in [(0,39),(0,10),(10,15),(15,20),(20,25),(25,39)]:
    DFx2=DFx1[(np.abs(DFx1.lat) >= r[0]) & (np.abs(DFx1.lat) < r[1])]

    DFx2 = DFx2[~DFx2['AR'].isin(ARout)].copy()
    print(f'range: {r} / ARs: {len(set(DFx2.AR))}')

    g=sns.scatterplot(data=DFx2,x='t_norm',y='value',hue='variable',alpha=0.2)
    DFx2=DFx2.groupby(['AR','variable','t_mean']).median().reset_index()
    g=sns.lineplot(
        data=DFx2,
        x='t_mean',
        y='value',
        hue='variable',
        estimator='median',
        errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
        err_style='band',   # o 'bars' para barras verticales
        marker='o'          # opcional para marcar puntos medios
    )
 #   sns.lineplot(data=DFx2.groupby(['t_norm','variable']).median().reset_index(),x='t_norm',y='value',hue='variable')

    plt.xlim(0,1)
    plt.ylim(-40,40)

    plt.ylabel(r'$\alpha$ [deg]')
    plt.xlabel(r'$t_{norm}$')

    plt.axhline(0,linestyle='dashed',color='gray')

    plt.title(fr'Latitude range: {r[0]}$^\circ -$  {r[1]}$^\circ$')

    handles, labels = g.get_legend_handles_labels()

 #   plt.legend(title='',handles=handles,labels=labels[:2])

    plt.legend(title='',handles=handles,labels=['Bayes','Barycenters'])
    plt.show()

# --- Nueva celda ---

for r in [(0,36),(0,10),(10,15),(15,20),(20,25),(25,36)]:

  #  resar=list(set(DF[DF.N0.abs()>0.6].AR.values))
    #resar=list(set(DF.AR.values))
    DFmax=DF.groupby('AR').max().reset_index()
    resar=list(set(DFmax[DFmax.flux>DFmax.flux.median()].AR.values))
    DFx2=DFx1[(np.abs(DFx1.lat) >= r[0]) & (np.abs(DFx1.lat) < r[1]) & (DFx1['AR'].isin(resar))]
    print(f'range: {r} / ARs: {len(set(DFx2.AR))}')

    g=sns.scatterplot(data=DFx2,x='t_norm',y='value',hue='variable',alpha=0.2)
    DFx2=DFx2.groupby(['AR','variable','t_mean']).mean().reset_index()
    g=sns.lineplot(
        data=DFx2,
        x='t_mean',
        y='value',
        hue='variable',
        estimator='median',
        errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
        err_style='band',   # o 'bars' para barras verticales
        marker='o'          # opcional para marcar puntos medios
    )
 #   sns.lineplot(data=DFx2.groupby(['t_norm','variable']).median().reset_index(),x='t_norm',y='value',hue='variable')

    plt.xlim(0,1)
    plt.ylim(-40,40)

    plt.ylabel(r'$\alpha$ [deg]')
    plt.xlabel(r'$t_{norm}$')

    plt.axhline(0,linestyle='dashed',color='gray')

    plt.title(fr'Latitude range: {r[0]}$^\circ -$  {r[1]}$^\circ$')

    handles, labels = g.get_legend_handles_labels()

 #   plt.legend(title='',handles=handles,labels=labels[:2])

    plt.legend(title='',handles=handles,labels=['Bayes','Barycenters'])
    plt.show()

# --- Nueva celda ---

DFx1

# --- Nueva celda ---

len(set(df_int.AR.values))

# --- Nueva celda ---

import numpy as np
import scipy.stats
from sklearn.metrics import r2_score, mean_squared_error


ftsz=10

def theil_sen_ci(x, y, n_boot=1000, ci=95):
    slopes = []
    intercepts = []
    rng = np.random.default_rng()
    for _ in range(n_boot):
        idx = rng.choice(len(x), len(x), replace=True)
        try:
            res = scipy.stats.theilslopes(y[idx], x[idx])
            slopes.append(res[0])
            intercepts.append(res[1])
        except Exception:
            continue
    lower = (100 - ci) / 2
    upper = 100 - lower
    slope_ci = np.percentile(slopes, [lower, upper])
    intercept_ci = np.percentile(intercepts, [lower, upper])
    return np.mean(slopes), np.mean(intercepts), slope_ci, intercept_ci

fig, axs = plt.subplots(1, 2, figsize=(8, 5)) # sharex=True, sharey=True)
axs = axs.ravel()
intervals = [(-0.1, 1), (0.8, 1)]

#DFxx = DFx1[~DFx1['AR'].isin(ARout)]
DFxx=DFx1

for i, (low, high) in enumerate(intervals):
    df_int = DFxx[(DFxx['t_norm'] > low) & (DFxx['t_norm'] <= high)]
    df_int['lat'] = df_int['lat'].apply(lambda x: np.abs(x))
    df_plot = df_int.groupby(['AR', 'variable','t_mean']).mean(numeric_only=True).reset_index()
    #sns.scatterplot(data=df_plot, x='lat', y='value', hue='variable', alpha=0.5, ax=axs[i])
    sns.regplot(data=df_plot[df_plot.variable=='alpha'], x='lat', y='value', x_bins=10,color='tab:blue', line_kws={'alpha':0.0},ax=axs[i])
    sns.regplot(data=df_plot[df_plot.variable=='alphab'], x='lat', y='value', x_bins=10,color='tab:orange', line_kws={'alpha':0.0},ax=axs[i])

    for var, color in zip(['alpha', 'alphab'], ['tab:blue', 'tab:orange']):
        dfv = df_plot[df_plot['variable'] == var]
        if len(dfv) > 1:
            slope, intercept, slope_ci, intercept_ci = theil_sen_ci(dfv['lat'].values, dfv['value'].values)
            x_fit = np.linspace(dfv['lat'].min(), dfv['lat'].max(), 100)
            y_fit = intercept + slope * x_fit
            y_fit_lower = intercept_ci[0] + slope_ci[0] * x_fit
            y_fit_upper = intercept_ci[1] + slope_ci[1] * x_fit
            axs[i].plot(x_fit, y_fit, color=color, label=f'Theil-Sen')
            axs[i].fill_between(x_fit, y_fit_lower, y_fit_upper, color=color, alpha=0.2)
            axs[i].text(
                0.05, 0.15 if var == 'alpha' else 0.05,
                f"a: {slope:.2f} ± {((slope_ci[1]-slope_ci[0])/2):.2f}",
                color=color,
                transform=axs[i].transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round", alpha=0.2, color=color)
            )

            axs[i].text(
                0.55, 0.15 if var == 'alpha' else 0.05,
                f"b: {intercept:.1f} ± {((intercept_ci[1]-intercept_ci[0])/2):.1f}",
                color=color,
                transform=axs[i].transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round", alpha=0.2, color=color)
            )

            axs[i].text(
                0.05, 0.25 ,
                r"$\alpha = \mathrm{a}~\theta +\mathrm{b}$",
                color='black',
                transform=axs[i].transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round", alpha=0.2, color='gray')
            )

            y_true = dfv['value'].values
            y_pred = intercept + slope * dfv['lat'].values
            r2 = r2_score(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred)
            slope, intercept, r_value, p_value, std_err = stats.linregress(dfv['lat'].values, dfv['value'].values)
            print(f'{var}--> R2: {r2:.2f}, RMSE: {rmse:.2f}, p-val:{p_value:.4f}, r:{r_value:.2f}')

    axs[i].tick_params(axis='both', labelsize=ftsz)  # Set y-ticklabel fontsize
    #g.ax_marg_y.tick_params(axis='x', labelsize=10)  # Set x-ticklabel fontsize (if needed)

    if i != 0:
        axs[i].legend().remove()
    axs[i].set_title(f"$t_{{norm}}$ in ({low}, {high}]")
    axs[i].axhline(0, color='gray', linestyle='dashed', linewidth=1)
    axs[i].set_xlabel(r'$\theta$ [deg]',fontsize=ftsz)
    axs[i].set_ylabel(r'$\alpha$ [deg]',fontsize=ftsz)




handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles=handles, labels=[r'$\alpha_\mathrm{mod}$', r'$\alpha_\mathrm{bar}$'], title='', loc='upper right')
#plt.ylim(-20,60)
plt.tight_layout()
#plt.savefig('./plots/joyplot.pdf',dpi=300)
plt.show()

# --- Nueva celda ---

ftsz=12

fig, axs = plt.subplots(1, 2, figsize=(9, 4),sharey=True) # sharex=True, sharey=True)
axs = axs.ravel()
intervals = [(0, 1), (0.8, 1)]

#DFxx = DFx1[~DFx1['AR'].isin(ARout)]
DFxx=DFx1

for i, (low, high) in enumerate(intervals):
    df_int = DFxx[(DFxx['t_norm'] >= low) & (DFxx['t_norm'] <= high)]
    df_int['lat'] = df_int['lat'].apply(lambda x: np.abs(x))
    df_plot = df_int.groupby(['AR', 'variable','t_mean']).mean(numeric_only=True).reset_index()
    #sns.scatterplot(data=df_plot, x='lat', y='value', hue='variable', alpha=0.5, ax=axs[i])
    sns.regplot(data=df_plot[df_plot.variable=='alpha'], x='lat', y='value', x_bins=10,color='tab:blue',
                #line_kws={'color':'red'},
                line_kws={'linestyle':'dashed'},
                ax=axs[i],label=r'$\alpha_\mathrm{mod}$')
    sns.regplot(data=df_plot[df_plot.variable=='alphab'], x='lat', y='value', x_bins=10,color='tab:orange',
                #line_kws={'color':'green'},
                line_kws={'linestyle':'dashed'},
                ax=axs[i],label=r'$\alpha_\mathrm{bar}$')

    axs[i].set_ylim(-5,20)
    axs[i].set_xlim(0,40)
    axs[i].tick_params(axis='both',labelsize=ftsz)

    axs[i].set_title(f"$t_{{norm}}$ in [{low}, {high}]")
    axs[i].axhline(0, color='gray', linestyle='dashed', linewidth=1)
    axs[i].set_xlabel(r'$\theta$ [deg]',fontsize=ftsz)
    if i==0:
        axs[i].set_ylabel(r'$\alpha$ [deg]',fontsize=ftsz)
    else:
        axs[i].set_ylabel('')

    for var, color in zip(['alpha', 'alphab'], ['tab:blue', 'tab:orange']):
      dfv = df_plot[df_plot['variable'] == var]
      res = stats.linregress(dfv['lat'].values, dfv['value'].values)

      axs[i].text(
          0.65, 0.21 if var == 'alpha' else 0.09,
          f"a: {res.slope:.2f} ± {res.stderr:.2f}\nb: {res.intercept:.1f} ± {res.intercept_stderr:.1f}",
          color=color,
          transform=axs[i].transAxes,
          fontsize=10,
          bbox=dict(boxstyle="round", alpha=0.2, color=color)
      )

   #   axs[i].text(
   #       0.75, 0.25 if var == 'alpha' else 0.15,
   #       f"b: {res.intercept:.1f} ± {res.intercept_stderr:.1f}",
   #       color=color,
   #       transform=axs[i].transAxes,
   #       fontsize=10,
   #       bbox=dict(boxstyle="round", alpha=0.2, color=color)
   #   )

      axs[i].text(
          0.65, 0.35 ,
          r"$\alpha = \mathrm{a}~\theta +\mathrm{b}$",
          color='black',
          transform=axs[i].transAxes,
          fontsize=10,
          bbox=dict(boxstyle="round", alpha=0.2, color='gray')
      )

plt.tight_layout()
#plt.savefig('./plots/joyplot-B.pdf',dpi=300)
plt.show()


# --- Nueva celda ---

DFx1.flux

# --- Nueva celda ---

import numpy as np
import scipy.stats
from sklearn.metrics import r2_score, mean_squared_error

def theil_sen_ci(x, y, n_boot=1000, ci=95):
    slopes = []
    intercepts = []
    rng = np.random.default_rng()
    for _ in range(n_boot):
        idx = rng.choice(len(x), len(x), replace=True)
        try:
            res = scipy.stats.theilslopes(y[idx], x[idx])
            slopes.append(res[0])
            intercepts.append(res[1])
        except Exception:
            continue
    lower = (100 - ci) / 2
    upper = 100 - lower
    slope_ci = np.percentile(slopes, [lower, upper])
    intercept_ci = np.percentile(intercepts, [lower, upper])
    return np.mean(slopes), np.mean(intercepts), slope_ci, intercept_ci

fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
axs = axs.ravel()
intervals = ['all','high', 'medium', 'low']

DFxx = DFx1[~DFx1['AR'].isin(ARout)]
DFxx=DFxx[DFxx.t_mean <= 1]
DFmax=DFxx.groupby(['AR','variable']).max().reset_index()
DFmax['B_bin']=pd.qcut(DFmax['flux'],q=3,labels=['low','medium','high'])
#DFmax['B_bin'] = pd.cut(DFmax['flux'], bins=3, labels=['low', 'medium', 'high']) # for equal lenght bins
DFxx = DFxx.merge(DFmax[['AR', 'variable', 'B_bin']], on=['AR', 'variable'], how='left')


for i, bb in enumerate(intervals):
    if bb == 'all':
        df_int = DFxx
    else:
        df_int = DFxx[DFxx.B_bin==bb]
    df_int['lat'] = df_int['lat'].apply(lambda x: np.abs(x))
    df_plot = df_int.groupby(['AR', 'variable','t_mean']).mean(numeric_only=True).reset_index()
    sns.scatterplot(data=df_plot, x='lat', y='value', hue='variable', alpha=0.5, ax=axs[i])

    for var, color in zip(['alpha', 'alphab'], ['tab:blue', 'tab:orange']):
        dfv = df_plot[df_plot['variable'] == var]
        if len(dfv) > 1:
            slope, intercept, slope_ci, intercept_ci = theil_sen_ci(dfv['lat'].values, dfv['value'].values)
            x_fit = np.linspace(dfv['lat'].min(), dfv['lat'].max(), 100)
            y_fit = intercept + slope * x_fit
            y_fit_lower = intercept_ci[0] + slope_ci[0] * x_fit
            y_fit_upper = intercept_ci[1] + slope_ci[1] * x_fit
            axs[i].plot(x_fit, y_fit, color=color, label=f'Theil-Sen')
            axs[i].fill_between(x_fit, y_fit_lower, y_fit_upper, color=color, alpha=0.2)
            axs[i].text(
                0.05, 0.15 if var == 'alpha' else 0.05,
                f"a: {slope:.2f} ± {((slope_ci[1]-slope_ci[0])/2):.2f}",
                color=color,
                transform=axs[i].transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round", alpha=0.2, color=color)
            )

            axs[i].text(
                0.55, 0.15 if var == 'alpha' else 0.05,
                f"b: {intercept:.1f} ± {((intercept_ci[1]-intercept_ci[0])/2):.1f}",
                color=color,
                transform=axs[i].transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round", alpha=0.2, color=color)
            )

            axs[i].text(
                0.05, 0.25 ,
                r"$\alpha = \mathrm{a}~\theta +\mathrm{b}$",
                color='black',
                transform=axs[i].transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round", alpha=0.2, color='gray')
            )
            axs[i].set_title(f"$\Phi_\mathrm{{max}}$ in [{df_int[df_int.t_mean==1].flux.min():.2e}, {df_int[df_int.t_mean==1].flux.max():.2e}] Mx")

            y_true = dfv['value'].values
            y_pred = intercept + slope * dfv['lat'].values
            r2 = r2_score(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred)
            slope, intercept, r_value, p_value, std_err = stats.linregress(dfv['lat'].values, dfv['value'].values)
            print(f'{var}--> R2: {r2:.2f}, RMSE: {rmse:.2f}, p-val:{p_value:.4f}')



    if i != 0:
        axs[i].legend().remove()

    axs[i].axhline(0, color='gray', linestyle='dashed', linewidth=1)
    axs[i].set_xlabel(r'$\theta$ [deg]')
    axs[i].set_ylabel(r'$\alpha$ [deg]')

handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles=handles, labels=[r'$\alpha_\mathrm{mod}$', r'$\alpha_\mathrm{bar}$'], title='', loc='upper right')
#plt.ylim(-20,60)
plt.savefig('./plots/joyplot-B.pdf',dpi=300)
plt.show()

# --- Nueva celda ---

len(df_int.AR.unique())

# --- Nueva celda ---

import numpy as np
import scipy.stats
from sklearn.metrics import r2_score, mean_squared_error

def theil_sen_ci(x, y, n_boot=1000, ci=95):
    slopes = []
    intercepts = []
    rng = np.random.default_rng()
    for _ in range(n_boot):
        idx = rng.choice(len(x), len(x), replace=True)
        try:
            res = scipy.stats.theilslopes(y[idx], x[idx])
            slopes.append(res[0])
            intercepts.append(res[1])
        except Exception:
            continue
    lower = (100 - ci) / 2
    upper = 100 - lower
    slope_ci = np.percentile(slopes, [lower, upper])
    intercept_ci = np.percentile(intercepts, [lower, upper])
    return np.mean(slopes), np.mean(intercepts), slope_ci, intercept_ci

fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
axs = axs.ravel()
intervals = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]

DFxx = DFx1[~DFx1['AR'].isin(ARout)]

for i, (low, high) in enumerate(intervals):
    df_int = DFxx[(DFxx['t_norm'] > low) & (DFxx['t_norm'] <= high)]
    df_int['lat'] = df_int['sin_lat'].apply(lambda x: np.abs(x))
    df_plot = df_int.groupby(['AR', 'variable']).mean(numeric_only=True).reset_index()
    sns.scatterplot(data=df_plot, x='sin_lat', y='value', hue='variable', alpha=0.5, ax=axs[i])

    for var, color in zip(['alpha', 'alphab'], ['tab:blue', 'tab:orange']):
        dfv = df_plot[df_plot['variable'] == var]
        if len(dfv) > 1:
            slope, intercept, slope_ci, intercept_ci = theil_sen_ci(dfv['sin_lat'].values, dfv['value'].values)
            x_fit = np.linspace(dfv['sin_lat'].min(), dfv['lat'].max(), 100)
            y_fit = intercept + slope * x_fit
            y_fit_lower = intercept_ci[0] + slope_ci[0] * x_fit
            y_fit_upper = intercept_ci[1] + slope_ci[1] * x_fit
            axs[i].plot(x_fit, y_fit, color=color, label=f'Theil-Sen')
            axs[i].fill_between(x_fit, y_fit_lower, y_fit_upper, color=color, alpha=0.2)
            axs[i].text(
                0.05, 0.15 if var == 'alpha' else 0.05,
                f"a: {slope:.2f} ± {((slope_ci[1]-slope_ci[0])/2):.2f}",
                color=color,
                transform=axs[i].transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round", alpha=0.2, color=color)
            )

            axs[i].text(
                0.55, 0.15 if var == 'alpha' else 0.05,
                f"b: {intercept:.1f} ± {((intercept_ci[1]-intercept_ci[0])/2):.1f}",
                color=color,
                transform=axs[i].transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round", alpha=0.2, color=color)
            )

            axs[i].text(
                0.05, 0.25 ,
                r"$\alpha = \mathrm{a}~\theta +\mathrm{b}$",
                color='black',
                transform=axs[i].transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round", alpha=0.2, color='gray')
            )

            y_true = dfv['value'].values
            y_pred = intercept + slope * dfv['sin_lat'].values
            r2 = r2_score(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred)
            print(f'{var}--> R2: {r2:.2f}, RMSE: {rmse:.2f}')



    if i != 0:
        axs[i].legend().remove()
    axs[i].set_title(f"$t_{{norm}}$ in ({low}, {high}]")
    axs[i].axhline(0, color='gray', linestyle='dashed', linewidth=1)
    axs[i].set_xlabel(r'$\sin(\theta)$')
    axs[i].set_ylabel(r'$\alpha$ [deg]')

handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles=handles, labels=[r'$\alpha_\mathrm{mod}$', r'$\alpha_\mathrm{bar}$'], title='', loc='upper right')

plt.tight_layout()
plt.show()

# --- Nueva celda ---

rmse

# --- Nueva celda ---

from scipy.stats import t

tinv = lambda p, df: abs(t.ppf(p/2, df))



for r in [(0,1),(0,0.25),(0.25,0.5),(0.5,0.75),(0.75,1),(1,2),(0.99,1.1)]:
  #  resar=list(set(DF[DF.N0<0].AR.values))
 #   resar=list(set(DF.AR.values))
#   DFmax=DF.groupby('AR').max().reset_index()
    DFx2=DFx1[(np.abs(DFx1.t_norm) >= r[0]) & (np.abs(DFx1.t_norm) < r[1])]
    DFx2=DFx2.groupby(['AR','variable']).median().reset_index()
    DFx2 = DFx2[~DFx2['AR'].isin(ARout)].copy()
    print(f'range: {r} / ARs: {len(set(DFx2.AR))}')

    #g=sns.scatterplot(data=DFx2.assign(lata=lambda x:np.abs(x.lat)),x='lata',y='value',hue='variable',alpha=0.3)
    sns.lmplot(data=DFx2.assign(lata=lambda x:np.abs(x.lat)),x='lata',y='value',hue='variable',x_bins=10)
    #g=sns.lineplot(
    #    data=DFx2.assign(lata=lambda x:np.abs(x.lat),latm=lambda x:round(5*np.abs(x.lat))/5),
    #    x='lata',
    #    y='value',
    #    hue='variable',
    #    estimator='median',
    #    errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
    #    err_style='band',   # o 'bars' para barras verticales
    #    marker='o'          # opcional para marcar puntos medios
    #)
 #   sns.lineplot(data=DFx2.groupby(['t_norm','variable']).median().reset_index(),x='t_norm',y='value',hue='variable')

    x=np.abs(DFx2[DFx2.variable=='alpha'].lat.values)
    y=DFx2[DFx2.variable=='alpha'].value.values

    ts = tinv(0.05, len(x)-2)

#    slope, intercept, rv, pv, err = stats.linregress(x, y)
    res = stats.linregress(x, y)

    textstr = fr"$\rho$ = {res.rvalue:.1f}"
    plt.text(0.1, 0.25, textstr, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round",color='tab:blue', alpha=0.2))

    # Agregar la pendiente y el coeficiente de correlación al gráfico
    textstr = fr"$\alpha$ = ({res.slope:.1f}$\pm$ {res.stderr:.1f})$\theta$ + ({res.intercept:.1f}$\pm$ {res.intercept_stderr:.1f})$^\circ$"
    plt.text(0.6, 0.3, textstr, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round",color='tab:blue', alpha=0.2))


    x=np.abs(DFx2[DFx2.variable=='alphab'].lat.values)
    y=DFx2[DFx2.variable=='alphab'].value.values

    ts = tinv(0.05, len(x)-2)

#    slope, intercept, rv, pv, err = stats.linregress(x, y)
    res = stats.linregress(x, y)

    # Agregar la pendiente y el coeficiente de correlación al gráfico
    textstr = fr"$\alpha$ = ({res.slope:.1f}$\pm$ {res.stderr:.1f})$\theta$ + ({res.intercept:.1f}$\pm$ {res.intercept_stderr:.1f})$^\circ$"
    plt.text(0.6, 0.2, textstr, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round",color='tab:orange', alpha=0.2))

    textstr = fr"$\rho$ = {res.rvalue:.1f}"
    plt.text(0.1, 0.15, textstr, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round",color='tab:orange', alpha=0.2))

    plt.xlim(0,35)
    plt.ylabel(r'$\alpha$ [deg]')
    plt.xlabel(r'Latitude [deg]')

    plt.axhline(0,linestyle='dashed',color='gray')

    plt.title(fr't_norm range: {r[0]} -- {r[1]}')

    handles, labels = g.get_legend_handles_labels()


    plt.legend(title='',handles=handles,labels=['Bayes','Barycenters'],loc='upper right')
    plt.show()

# --- Nueva celda ---

from scipy.stats import t

tinv = lambda p, df: abs(t.ppf(p/2, df))



for r in [(0,1),(0,0.25),(0.25,0.5),(0.5,0.75),(0.75,1),(1,2),(0.99,1.1)]:
    DFx2=DFx1[(np.abs(DFx1.t_norm) >= r[0]) & (np.abs(DFx1.t_norm) < r[1])]
    DFx2=DFx2.groupby(['AR','variable']).median().reset_index()
    print(f'range: {r} / ARs: {len(set(DFx2.AR))}')

    #g=sns.scatterplot(data=DFx2.assign(lata=lambda x:np.abs(x.lat)),x='lata',y='value',hue='variable',alpha=0.3)
    sns.lmplot(data=DFx2.assign(lata=lambda x:np.sin(np.abs(np.pi*x.lat/180))),x='lata',y='value',hue='variable',x_bins=10)
    #g=sns.lineplot(
    #    data=DFx2.assign(lata=lambda x:np.abs(x.lat),latm=lambda x:round(5*np.abs(x.lat))/5),
    #    x='lata',
    #    y='value',
    #    hue='variable',
    #    estimator='median',
    #    errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
    #    err_style='band',   # o 'bars' para barras verticales
    #    marker='o'          # opcional para marcar puntos medios
    #)
 #   sns.lineplot(data=DFx2.groupby(['t_norm','variable']).median().reset_index(),x='t_norm',y='value',hue='variable')

    x=np.sin(np.pi*np.abs(DFx2[DFx2.variable=='alpha'].lat.values/180))
    y=DFx2[DFx2.variable=='alpha'].value.values

    ts = tinv(0.05, len(x)-2)

#    slope, intercept, rv, pv, err = stats.linregress(x, y)
    res = stats.linregress(x, y)

    textstr = fr"$\rho$ = {res.rvalue:.1f}"
    plt.text(0.1, 0.25, textstr, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round",color='tab:blue', alpha=0.2))

    # Agregar la pendiente y el coeficiente de correlación al gráfico
    textstr = fr"$\alpha$ = ({res.slope:.1f}$\pm$ {res.stderr:.1f})$\theta$ + ({res.intercept:.1f}$\pm$ {res.intercept_stderr:.1f})$^\circ$"
    plt.text(0.6, 0.3, textstr, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round",color='tab:blue', alpha=0.2))


    x=np.sin(np.pi*np.abs(DFx2[DFx2.variable=='alphab'].lat.values/180))
    y=DFx2[DFx2.variable=='alphab'].value.values

    ts = tinv(0.05, len(x)-2)

#    slope, intercept, rv, pv, err = stats.linregress(x, y)
    res = stats.linregress(x, y)

    # Agregar la pendiente y el coeficiente de correlación al gráfico
    textstr = fr"$\alpha$ = ({res.slope:.1f}$\pm$ {res.stderr:.1f})$\theta$ + ({res.intercept:.1f}$\pm$ {res.intercept_stderr:.1f})$^\circ$"
    plt.text(0.6, 0.2, textstr, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round",color='tab:orange', alpha=0.2))

    textstr = fr"$\rho$ = {res.rvalue:.1f}"
    plt.text(0.1, 0.15, textstr, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round",color='tab:orange', alpha=0.2))

    #plt.xlim(0,35)
    plt.ylabel(r'$\alpha$ [deg]')
    plt.xlabel(r'Latitude [deg]')

    plt.axhline(0,linestyle='dashed',color='gray')

    plt.title(fr't_norm range: {r[0]} -- {r[1]}')

    handles, labels = g.get_legend_handles_labels()


    plt.legend(title='',handles=handles,labels=['Bayes','Barycenters'],loc='upper right')
    plt.show()

# --- Nueva celda ---

DF=DF.assign(difalpha=lambda x: x.alpha-x.alphab)

# --- Nueva celda ---

DFtest=DF[DF.AR.isin(ars2)]

# --- Nueva celda ---

list(set(DFtest[DFtest.alpha < -50].AR))

# --- Nueva celda ---




# --- Nueva celda ---

name=8016
#print(DFx1[DFx1.AR ==name])
for name in list(set(DFtest[DFtest.alpha > 60].AR)):
    sns.lineplot(DFx1[DFx1.AR ==name],x='t_norm',y='value',hue='variable',marker='o' )
    sns.lineplot(DF[DF.AR ==name],x='fint',y='alpha',marker='o' ,color='black')
    plt.title(name)
    plt.show()
#sns.lineplot(DFx1[DFx1.AR ==name],x='t_norm',y='fn',marker='o' )


# --- Nueva celda ---



# --- Nueva celda ---

DFx1['t_norm']=DFx1.t_norm.apply(lambda x: f'{x:.1f}')

# --- Nueva celda ---

sns.boxplot(data=DFx1,x='t_norm',y='value',hue='variable',showfliers=False)

# --- Nueva celda ---

DFx1.f_grid

# --- Nueva celda ---

len(tilta)

# --- Nueva celda ---

tilt_meana = np.nanmedian(tilta, axis=0)
tilt_stda = np.nanstd(tilta, axis=0)

tilt_meanb = np.nanmedian(tiltb, axis=0)
tilt_stdb = np.nanstd(tiltb, axis=0)

# --- Nueva celda ---

plt.plot(np.linspace(0,1,11), tilt_meana, label='Media del tilt')
plt.fill_between(np.linspace(0,1,11), tilt_meana - tilt_stda, tilt_meana + tilt_stda, alpha=0.3, label='±1σ')

plt.plot(np.linspace(0,1,11), tilt_meanb, label='Media del tilt')
plt.fill_between(np.linspace(0,1,11), tilt_meanb - tilt_stdb, tilt_meanb + tilt_stdb, alpha=0.3, label='±1σ')

# --- Nueva celda ---

DF[DF.AR == 8913]

# --- Nueva celda ---

# Convertir a array 2D: (n_regiones, n_fases)
tilts_array = np.array(tilts_interp_all)

# Calcular media y std (ignorando NaNs)
tilt_mean = np.nanmean(tilts_array, axis=0)
tilt_std = np.nanstd(tilts_array, axis=0)

# --- Nueva celda ---

DF[DF.AR == 8913]

# --- Nueva celda ---



# --- Nueva celda ---



# --- Nueva celda ---

fig, axs = plt.subplots(3, 1, figsize=(6, 12),sharex=True)



for en,ra in enumerate([(0,12),(12,20),(20,45)]):
    axs[en].set_ylim(-5,5)

    axs[en].axhline(0,color='black')

    sns.scatterplot(data=DF[(DF.lat.abs() >=ra[0]) & (DF.lat.abs() <ra[1])],ax=axs[en],x='fint',y='rot',alpha=0.5,s=30)
    sns.scatterplot(data=DF[(DF.lat.abs() >=ra[0]) & (DF.lat.abs() <ra[1])],ax=axs[en],x='fint',y='rotb',alpha=0.5,s=30)

    sns.regplot(data=DF[(DF.lat.abs() >=ra[0]) & (DF.lat.abs() <ra[1])],ax=axs[en],x='fint',y='rot',color='tab:red',x_bins=10,label='Bayesian')
    sns.regplot(data=DF[(DF.lat.abs() >=ra[0]) & (DF.lat.abs() <ra[1])],ax=axs[en],x='fint',y='rotb',color='tab:green',x_bins=10,label='Barycenters')
    axs[en].legend()

    axs[en].text(.7,.1,rf'${ra[0]}^\circ<|lat|\leq {ra[1]}^\circ$', transform=axs[en].transAxes, fontsize=12,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round',alpha=0.5))
#    plt.set_xlabel('Normalized flux')
    axs[en].set_ylabel(r'$\Delta \alpha$ [deg/hour]')

fig.tight_layout(pad=1.0)

axs[2].set_xlabel('Normalized flux')



# --- Nueva celda ---



fig, axs = plt.subplots(2, 2, figsize=(10, 7),sharey=True,sharex=True)

axs=np.ravel(axs)

DF2x=DF2[(DF2.lat.abs() >=20) & (DF2.lat.abs() <45)]


for d in [1,2,3,4]:


    DFx=DF2x[(DF2x['frange']==d)]

    sns.histplot(data=DFx,x='rot',alpha=0.5,bins=10,binrange=(-2,2),label='Bayesian',ax=axs[d-1])

    sns.histplot(data=DFx,x='rotb',alpha=0.5,bins=10,binrange=(-2,2),label='Barycenters',ax=axs[d-1])

   # sns.histplot(data=DFx,x='ratio',alpha=0.5,bins=10,binrange=(-60,60),label='Model-Barycenters')



    axs[d-1].axvline(0,color='black',linestyle='dashed')

    axs[d-1].axvline(DFx.rot.mean(),color='tab:blue')
    axs[d-1].axvline(DFx.rotb.mean(),color='tab:orange')
    axs[d-1].text(0.05, 0.75, f"                     \n                 ",
             transform=axs[d-1].transAxes,
             fontsize=12,
             bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round',alpha=0.2))

    mayores_cero=[]
    menores_cero=[]
    for en,at in enumerate(['rot','rotb']):
        # Calcular los valores mayores y menores que cero
        mayores_cero.append((DFx[at] > 0).sum())
        menores_cero.append((DFx[at] < 0).sum())

    axs[d-1].text(0.05, 0.75, f"α > 0:\nα < 0:", transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.18, 0.75, f"{mayores_cero[0]} \n{menores_cero[0]}", color='tab:blue', transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.24, 0.75, f"{mayores_cero[1]} \n{menores_cero[1]}", color='tab:orange', transform=axs[d-1].transAxes, fontsize=12)


    axs[d-1].text(0.75, 0.65, f"                ",
         transform=axs[d-1].transAxes,
         fontsize=12,
         bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round',alpha=0.2))

    axs[d-1].text(0.75, 0.65, r"$\bar{\alpha}:$", transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.8, 0.65, f"{DFx.rot.mean():.1f}", color='tab:blue', transform=axs[d-1].transAxes, fontsize=12)
    axs[d-1].text(0.87, 0.65, f"{DFx.rotb.mean():.1f}", color='tab:orange', transform=axs[d-1].transAxes, fontsize=12)

       # axs[d-1].text(0.05, 0.80, f"α < 0: ,{menores_cero[1]}", transform=axs[d-1].transAxes, fontsize=12)

        # Agregar texto con la información
        # Agregar texto con formato LaTeX
#    axs[d-1].text(
#    0.05, 0.85-en/5,
 #   rf"$\alpha > 0: \color{{blue}}{{mayores_cero[0]}},{mayores_cero[1]}$"+f"\n"+
 #   rf"$\alpha < 0: {menores_cero[0]},{menores_cero[1]}$",
 #   transform=axs[d-1].transAxes,
 #   fontsize=12,




axs[1].legend()

axs[0].set_ylabel('ARs')
axs[2].set_ylabel('ARs')

axs[2].set_xlabel(r'$\Delta \alpha$ [deg/hour]')
axs[3].set_xlabel(r'$\Delta \alpha$ [deg/hour]')


fig.tight_layout(pad=1.0)

fig.show()

# --- Nueva celda ---



fig, axs = plt.subplots(2, 2, figsize=(10, 7),sharey=True,sharex=True)

axs=np.ravel(axs)

for d in [1,2,3,4]:


    DFx=DF2[DF2['frange']==d]

    sns.histplot(data=DFx,x='rot',alpha=0.5,bins=10,binrange=(-2,2),label='Bayesian Model',ax=axs[d-1])

    sns.histplot(data=DFx,x='rotb',alpha=0.5,bins=10,binrange=(-2,2),label='Barycenters',ax=axs[d-1])

   # sns.histplot(data=DFx,x='ratio',alpha=0.5,bins=10,binrange=(-60,60),label='Model-Barycenters')



    axs[d-1].axvline(0,color='black',linestyle='dashed')

    axs[d-1].axvline(DFx.rot.mean(),color='tab:blue')
    axs[d-1].axvline(DFx.rotb.mean(),color='tab:orange')

    for en,at in enumerate(['rot','rotb']):
        # Calcular los valores mayores y menores que cero
        mayores_cero = (DFx[at] > 0).sum()
        menores_cero = (DFx[at] < 0).sum()

        # Agregar texto con la información
        # Agregar texto con formato LaTeX
        axs[d-1].text(
        0.05, 0.85-en/5,
        rf"$\Delta \alpha > 0: {mayores_cero}$"+f"\n"+
        rf"$\Delta \alpha < 0: {menores_cero}$",
        transform=axs[d-1].transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle="round", alpha=0.2,color=plt.cm.get_cmap('tab10')(en)))


    axs[d-1].legend()

axs[0].set_ylabel('ARs')
axs[2].set_ylabel('ARs')

axs[2].set_xlabel(r'$\Delta \alpha / \Delta t$ [deg/hour]')
axs[3].set_xlabel(r'$\Delta \alpha / \Delta t$ [deg/hour]')


fig.tight_layout(pad=1.0)

# --- Nueva celda ---

fig, axs = plt.subplots(2, 2, figsize=(10, 7),sharey=True,sharex=True)

axs=np.ravel(axs)

for d in [1,2,3,4]:


    DFx=DF2[DF2['frange']==d]
    DFx=DFx.assign(ratea=lambda x: x.alpha*x.rot)
    DFx=DFx.assign(rateb=lambda x: x.alphab*x.rotb)

    sns.histplot(data=DFx,x='ratea',alpha=0.5,bins=10,binrange=(-8,8),label='Bayesian Model',ax=axs[d-1])

    sns.histplot(data=DFx,x='rateb',alpha=0.5,bins=10,binrange=(-8,8),label='Barycenters',ax=axs[d-1])

   # sns.histplot(data=DFx,x='ratio',alpha=0.5,bins=10,binrange=(-60,60),label='Model-Barycenters')



    axs[d-1].axvline(0,color='black',linestyle='dashed')

   # axs[d-1].axvline(DFx.ratea.mean(),color='tab:blue')
   # axs[d-1].axvline(DFx.rateb.mean(),color='tab:orange')

    for en,at in enumerate(['ratea','rateb']):
        # Calcular los valores mayores y menores que cero
        mayores_cero = (DFx[at] > 0).sum()
        menores_cero = (DFx[at] < 0).sum()

        # Agregar texto con la información
        # Agregar texto con formato LaTeX
        axs[d-1].text(
        0.05, 0.85-en/5,
        rf"$\alpha\Delta \alpha > 0: {mayores_cero}$"+f"\n"+
        rf"$\alpha\Delta \alpha < 0: {menores_cero}$",
        transform=axs[d-1].transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle="round", alpha=0.2,color=plt.cm.get_cmap('tab10')(en)))


    axs[d-1].legend()

axs[0].set_ylabel('ARs')
axs[2].set_ylabel('ARs')

axs[2].set_xlabel(r'$\alpha \Delta \alpha / \Delta t$ [deg**2/hour]')
axs[3].set_xlabel(r'$\alpha \Delta \alpha / \Delta t$ [deg**2/hour]')


fig.tight_layout(pad=1.0)

# --- Nueva celda ---

fig, axs = plt.subplots(2, 2, figsize=(10, 7),sharey=True,sharex=True)

axs=np.ravel(axs)

for d in [1,2,3,4]:


    DFx=DF2[DF2['frange']==d]
    DFx=DFx.assign(ratea=lambda x: x.rot/x.alpha)
    DFx=DFx.assign(rateb=lambda x: x.rotb/x.alphab)

    sns.histplot(data=DFx,x='ratea',alpha=0.5,bins=20,binrange=(-1,1),label='Bayesian Model',ax=axs[d-1])

    sns.histplot(data=DFx,x='rateb',alpha=0.5,bins=20,binrange=(-1,1),label='Barycenters',ax=axs[d-1])

   # sns.histplot(data=DFx,x='ratio',alpha=0.5,bins=10,binrange=(-60,60),label='Model-Barycenters')



    axs[d-1].axvline(0,color='black',linestyle='dashed')

   # axs[d-1].axvline(DFx.ratea.mean(),color='tab:blue')
   # axs[d-1].axvline(DFx.rateb.mean(),color='tab:orange')

    for en,at in enumerate(['ratea','rateb']):
        # Calcular los valores mayores y menores que cero
        mayores_cero = (DFx[at] > 0).sum()
        menores_cero = (DFx[at] < 0).sum()

        # Agregar texto con la información
        # Agregar texto con formato LaTeX
        axs[d-1].text(
        0.05, 0.85-en/5,
        rf"$\Delta \alpha / \alpha > 0: {mayores_cero}$"+f"\n"+
        rf"$\Delta \alpha / \alpha < 0: {menores_cero}$",
        transform=axs[d-1].transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle="round", alpha=0.2,color=plt.cm.get_cmap('tab10')(en)))


    axs[d-1].legend()

axs[0].set_ylabel('ARs')
axs[2].set_ylabel('ARs')

axs[2].set_xlabel(r'$\Delta \alpha / (\alpha \Delta t)$ [1/hour]')
axs[3].set_xlabel(r'$\Delta \alpha / (\alpha \Delta t)$ [1/hour]')


fig.tight_layout(pad=1.0)

# --- Nueva celda ---

sns.regplot(data=DF,y='alpha',x='fint', x_estimator=np.mean,label='Bayesian')
sns.regplot(data=DF,y='alphab',x='fint', x_estimator=np.mean,label='Barycenters')


plt.legend()
plt.ylabel(r'$\alpha$ [deg]')
plt.xlabel(r'Normalized flux')


# --- Nueva celda ---

sns.regplot(data=DF,y='rot',x='fint',x_bins=[0.25,0.5,0.75,1],label='Bayesian')
sns.regplot(data=DF,y='rotb',x='fint',x_bins=[0.25,0.5,0.75,1],label='Barycenters')


plt.legend()
plt.ylabel(r'$\Delta\alpha/\Delta t$ [deg/h]')
plt.xlabel(r'Normalized flux')


# --- Nueva celda ---

df_melt = DF2[['alpha','alphab','frange']].melt(['frange'], var_name='category', value_name='values')

# --- Nueva celda ---

g=sns.boxplot(data=df_melt,x='frange',y='values',hue='category')
plt.gca().xaxis.set_ticklabels([0.25,0.5,0.75,1])

plt.ylabel(r'$\alpha$ [deg]')

# Edit legend title and labels
handles, labels = g.get_legend_handles_labels()
g.legend(handles, ["Bayesian", "Barycenters"], title=" ")

# --- Nueva celda ---

df_melt = DF2[['rot','rotb','frange']].melt(['frange'], var_name='category', value_name='values')
g=sns.boxplot(data=df_melt,x='frange',y='values',hue='category',showfliers=False)
plt.gca().xaxis.set_ticklabels([0.25,0.5,0.75,1])

plt.ylabel(r'$\Delta\alpha/\Delta t$ [deg/h]')

# Edit legend title and labels
handles, labels = g.get_legend_handles_labels()
g.legend(handles, ["Bayesian", "Barycenters"], title=" ")

# --- Nueva celda ---

DF2[DF2.AR == 10879][['alpha','alphab','lat','frange']]

# --- Nueva celda ---

plt.plot(DF[DF.AR == 10879].alpha*180/np.pi)
plt.show()
plt.plot(DF2[DF2.AR == 10879].fint)
plt.show()
plt.plot(DF[DF.AR == 10879].alpha*180/np.pi)
plt.show()

# --- Nueva celda ---

plt.plot(DF[DF.AR == 10268].alpha*180/np.pi)



# --- Nueva celda ---

DF[(DF.fint > 0) & (DF.fint <= 0.25)].groupby(['AR']).mean()

# --- Nueva celda ---

