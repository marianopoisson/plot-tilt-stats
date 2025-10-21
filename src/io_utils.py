import pandas as pd
import numpy as np
from diptest import diptest
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from scipy import stats




def load_data(path):
    return pd.read_csv(path)

def save_figure(fig, path):
    fig.savefig(path, bbox_inches='tight')

def correct_df(df):
    df[df.AR == 8913]=df[df.AR == 8913].assign(fint=df[df.AR == 8913].flux/np.max(df[df.AR == 8913].flux))
    df['alpha']=df.apply(lambda x: -1*180*np.sign(x.lat)*x.alpha/np.pi,axis=1)
    df['alphab']=df.apply(lambda x: -1*np.sign(x.lat)*x.alphab,axis=1)
    df=df.assign(rot= lambda x: np.gradient(x.alpha,x.mag)/1.5)
    df=df.assign(rotb= lambda x: np.gradient(x.alphab,x.mag)/1.5)
    pixsize=1.98*0.725
    df=df.assign(cond1=df.a/(df.a+df.R))
    df['gamma'] = 0

    # Asignar el valor calculado solo a las filas que cumplen la condición (A > 10)
    df.loc[df['da']>= df['cond1'], 'gamma'] = 180*np.arccos((df.R+df.a)*(1-df.da)/df.R)/np.pi

    df=df.assign(sepax=lambda x: 2*pixsize*x.R*np.sin(np.pi*x.gamma/180))
    df['sar']=df['sar'].apply(lambda x: pixsize*x)

    return df

def filter_rotation(DF,threshold=4):
    ars=list(set(DF.AR.values))
    DFmax=np.abs(DF).groupby('AR').max().reset_index()
    listar=list(DFmax[np.abs(DFmax.rot) >threshold].AR)
    DFxx=[]

    for name in listar:
        DF2 = DF[DF.AR == name]

        limax=DF2[np.abs(DF2.rot)>threshold].max().mag
        DFmax=np.abs(DF2).groupby('AR').max().reset_index()

        if (np.abs(DFmax.rot.values) >4) & (DF2[DF2.mag==limax].fint.values <=0.3):
            print(name)
          #  sns.lineplot(data=DF2,x='fint',y='alpha')
          #  sns.lineplot(data=DF2[DF2.mag>limax],x='fint',y='alpha')

        # sns.lineplot(data=DF2,x='fint',y='alphab')
            plt.show()

            DFxx.append(DF2[DF2.mag>limax])

    DFxx=pd.concat(DFxx)

    ars2=list(set(listar)^set(ars))
    DFb=pd.concat([DFxx,DF[DF['AR'].isin(ars2)]])
    print(f'total ARs = {len(list(set(DFb.AR.values)))}')
    return DFb    



def norm_time(DF):

    DFx = []
    for name in list(set(DF.AR.values)):
        DF2 = DF[DF.AR == name]
        lower_lim =   DF2.fint.min()+0.2 
        # Normalización basada en el valor de mag en el punto de máximo flujo normalizado
        idx_max = DF2.fint.idxmax()
        magmax = DF2.loc[idx_max].mag
        # (opcional) linealizar si se desea ajustar f_norm como función de mag/magmax
        DF2p = DF2[DF2.fint < lower_lim]
        x_pre = DF2p.mag.values / magmax
        y_pre = DF2p.fint.values
        slope, intercept, _, _, _ = stats.linregress(y_pre, x_pre)
        # Definir eje de tiempo normalizado
        t_norm = (DF2.mag.values/ magmax - intercept) / (1 - intercept)

        DFx.append(pd.DataFrame({'AR':name,'alpha':DF2.alpha.values,'alphab':DF2.alphab.values,'t_norm':t_norm,
                             'lat':DF2.lat.mean(),'fn':DF2.fint.values,'Nt':DF2.N0.values,
                             'sar':DF2.sar.values,'sepax':DF2.sepax.values,
                             'mag':DF2.mag.values,'flux':DF2.flux.values,'fint':DF2.fint.values}))

    DFx=pd.concat(DFx)    
    return DFx



def check_multimodal(DF):
    ars=list(set(DF.AR.values))
    multiars=[]
    DFc=[]
    DFu=[]
    DFl=[]
    DFs=[]
    for name in ars:

        try:
            DF1=pd.read_csv('../data/posteriors2/'+str(name)+'_TM3.csv')
        except:
            DF1=pd.read_csv('../data/posteriors/'+str(name)+'_TM3.csv')

        for i in list(set(DF1['mag'].values)):
            data=DF1[DF1['mag']==i]['alpha'].values
            # Aplicar prueba de dip
            dip_stat, p_value = diptest(data)
        #  print(f"Estadístico de dip: {dip_stat}, p-valor: {p_value}")
            if p_value < 0.05:
                multiars.append(name)
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
        DFc=pd.concat(DFc)
        DFu=pd.concat(DFu)
        DFl=pd.concat(DFl)
        DFs=pd.concat(DFs)
    
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

    for name in set(multiars):
        DFb=pd.concat([DFs1[DFs1.AR==name],DFc[DFc.AR==name]]).sort_values(by='mag', ascending=True)[DFs.keys()[1:-1]].reset_index(drop=True)
        DFb.to_csv(f'../data/{name}_TM3_C.csv')