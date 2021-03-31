import sys
import logging
import traceback
from functools import wraps

import numpy as np
# import dask
# import dask_cudf
import dash

from typing import Union

logger = logging.getLogger(__name__)


# DELAYED_DF_TYPES = Union[dask.dataframe.core.DataFrame, dask_cudf.core.DataFrame]


def generate_colors(num_colors):
    """
    Generates evenly disributed colors
    """
    a = ((np.random.random(size=num_colors) * 255))
    b = ((np.random.random(size=num_colors) * 255))
    return [
        "#%02x%02x%02x" % (int(r), int(g), 125) for r, g in zip(a, b)
        ]


def report_ui_error(num_returns):
    """
    Excepts all exceptions from the wrapped function and manages the error msg
    for UI. The error msg is always added as the last return value. All other
    return values are set to dash.no_update. The function decorator needs to
    pass number of return values.
    """

    def _report_ui_error(func):

        @wraps(func)
        def func_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except dash.exceptions.PreventUpdate as e:
                raise e
            except Exception as e:
                traceback.print_exception(*sys.exc_info())
                ret_value = [dash.no_update for i in range(num_returns - 1)]
                ret_value.append(str(e))
                return ret_value
        return func_wrapper
    return _report_ui_error
