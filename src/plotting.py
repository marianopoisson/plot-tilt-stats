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
import glob as glob
import seaborn as sns
from scipy.stats import linregress
from scipy import stats
import os
from IPython.display import display, HTML
from diptest import diptest
from diptest import diptest
from diptest import diptest
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import numpy as np
import scipy.stats
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import scipy.stats
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import scipy.stats
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import t
from scipy.stats import t

import matplotlib.pyplot as plt

def set_plot_style():
    #plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'figure.figsize': (6,4),
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'font.size': 14
    })

def plot_comparison(data, vary1, vary2, title=None, save_path=None,all=None):
    set_plot_style()
    fig, ax = plt.subplots()
    if all:
        sns.scatterplot(data=data,x='t_norm',y=vary1,alpha=0.2,ax=ax,color='tab:blue')
        sns.scatterplot(data=data,x='t_norm',y=vary2,alpha=0.2,ax=ax,color='tab:orange')

    DFx2=data.groupby(['AR','t_mean']).mean().reset_index()
    sns.lineplot(
    data=DFx2,
    x='t_mean',
    y=vary1,
    estimator='mean',
    errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
    err_style='band',   # o 'bars' para barras verticales
    marker='o',          # opcional para marcar puntos medios
    ax=ax,color='tab:blue')
    
    sns.lineplot(
    data=DFx2,
    x='t_mean',
    y=vary2,
    estimator='mean',
    errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
    err_style='band',   # o 'bars' para barras verticales
    marker='o',          # opcional para marcar puntos medios
    ax=ax,color='tab:orange')
    if title:
        ax.set_title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig, ax



def plot_single(data, vary, title=None, save_path=None):
    set_plot_style()
    fig, ax = plt.subplots()
    sns.scatterplot(data=data,x='t_norm',y=vary,alpha=0.2,ax=ax)

    DFx2=data.groupby(['AR','t_mean']).mean().reset_index()
    sns.lineplot(
    data=DFx2,
    x='t_mean',
    y=vary,
    estimator='mean',
    errorbar='sd',      # 'sd' para desviación estándar, o 'ci' para intervalo de confianza
    err_style='band',   # o 'bars' para barras verticales
    marker='o',          # opcional para marcar puntos medios
    ax=ax)
    if title:
        ax.set_title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig, ax
