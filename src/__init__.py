"""
src package for the plot_tilt_stat project.

Provides modules for:
- io_utils: manejo de datos y utilidades
- analysis: análisis estadístico y filtrado
- plotting: funciones de graficado
- main: flujo principal del proyecto
"""

from . import io_utils
from . import analysis
from . import plotting

# Exponer funciones comunes al importar src directamente
from .io_utils import load_data, save_figure,correct_df,filter_rotation,norm_time       
from .analysis import compute_statistics, correlation_matrix
from .plotting import plot_comparison, set_plot_style,plot_single

__all__ = [
    "io_utils",
    "analysis",
    "plotting",
    "load_data",
    "save_figure",
    "compute_statistics",
    "correlation_matrix",
    "plot_comparison",
    "set_plot_style",
]
