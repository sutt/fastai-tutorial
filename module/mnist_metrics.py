import os, sys
import copy as copyroot
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
from fastai2.basics import *
from fastai2.vision.all import *

from sklearn.metrics import r2_score, mean_absolute_error

def metrics_df(
        learn,
        s_model,
        s_details,
        ):

    v_preds = learn.get_preds()
    t_preds = learn.get_preds(ds_idx=0)

    v_mse = mse(v_preds[0],v_preds[1]).item()
    t_mse = mse(t_preds[0], t_preds[1]).item()

    v_r2 = r2_score(v_preds[1].squeeze(1), v_preds[0])
    t_r2 = r2_score(t_preds[1].squeeze(1), t_preds[0])

    v_mae = mean_absolute_error(v_preds[1].squeeze(1), v_preds[0])
    t_mae = mean_absolute_error(t_preds[1].squeeze(1), t_preds[0])

    df = pd.DataFrame({
            'split':  ['valid','train'],
            'mse':    [v_mse, t_mse],
            'mae':    [v_mae, t_mae],
            'r2':     [v_r2, t_r2],
    })

    df['model'] = s_model

    df['details'] = s_details

    start_cols = ['model', 'details', 'split']

    col_order = (start_cols + [col for col in df.columns 
                                if col not in start_cols])
    df = df.loc[:,col_order]

    return df