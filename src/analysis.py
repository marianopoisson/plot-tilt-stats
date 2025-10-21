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

import numpy as np
import pandas as pd

def compute_statistics(df, cols):
    """Compute mean and std for given columns."""
    return df[cols].agg(['mean', 'std']).T

def correlation_matrix(df, cols):
    return df[cols].corr()
