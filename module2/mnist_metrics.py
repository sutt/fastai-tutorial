import os, sys
import copy as copyroot
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
from fastai.basics import *
from fastai.vision.all import *

from sklearn.metrics import r2_score, mean_absolute_error

def calc_dist(pred, actual):
    ''' these calcs are for ImagePoints, using cartesian distance '''

    dist     = ((pred - actual)**2).sum(1)**0.5
    baseline = ((actual - actual.mean(0))**2).sum(1)**0.5
    
    dist_avg    = dist.mean().item()
    dist_r2     = 1 - (dist.sum() / baseline.sum()).item()
    sqdist_avg  = (dist**2).mean().item()
    sqdist_r2   = 1 - ((dist**2).sum() / (baseline**2).sum()).item()

    return (dist_avg, dist_r2, sqdist_avg, sqdist_r2)


def metrics_df(
        learn,
        s_model,
        s_details,
        s_target,
        ):

    v_preds = learn.get_preds()
    t_preds = learn.get_preds(ds_idx=0)

    v_mse = mse(v_preds[0], v_preds[1]).item()
    t_mse = mse(t_preds[0], t_preds[1]).item()

    v_r2 = r2_score(v_preds[1].squeeze(1), v_preds[0])
    t_r2 = r2_score(t_preds[1].squeeze(1), t_preds[0])

    v_mae = mean_absolute_error(v_preds[1].squeeze(1), v_preds[0])
    t_mae = mean_absolute_error(t_preds[1].squeeze(1), t_preds[0])

    v_dist_avg, v_dist_r2, v_sqdist_avg, v_sqdist_r2 = calc_dist(
                                        v_preds[0], v_preds[1].squeeze(1)
                                        )
    
    t_dist_avg, t_dist_r2, t_sqdist_avg, t_sqdist_r2 = calc_dist(
                                        t_preds[0], t_preds[1].squeeze(1)
                                        )

    df = pd.DataFrame({
            'split':      ['valid','train'],
            'mse':        [v_mse, t_mse],
            'mae':        [v_mae, t_mae],
            'r2':         [v_r2, t_r2],
            'dist_avg':   [v_dist_avg, t_dist_avg],
            'dist_r2':    [v_dist_r2, t_dist_r2], 
            'sqdist_avg': [v_sqdist_avg, t_sqdist_avg], 
            'sqdist_r2':  [v_sqdist_r2,  t_sqdist_r2],
    })

    df['model'] = s_model
    df['details'] = s_details
    df['target'] = s_target

    start_cols = ['model', 'details', 'target', 'split']
    
    col_order = (start_cols + [col for col in df.columns 
                                if col not in start_cols])
    df = df.loc[:,col_order]

    return df