import os, sys
import copy as copyroot
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt

from fastai2.basics import *
from fastai2.vision.all import *

from module.mnist_helpers import build_df, eda_fig_1

path = untar_data(URLs.MNIST_TINY)
df = build_df(path)
df.head(2)

# works -----

db =   DataBlock(blocks=(ImageBlock(cls=PILImageBW), 
                         PointBlock), 
                splitter=RandomSplitter(),
                get_x=ColReader('fn', pref=path),
                get_y=ColReader(['point_topleft_x', 'point_topleft_y'])
                )

dl = db.dataloaders(df)

# breaks ---------

# db =   DataBlock(blocks=(ImageBlock(cls=PILImageBW), 
#                          PointBlock), 
#                 splitter=RandomSplitter(),
#                 get_x=ColReader('fn', pref=path),
#                 )

# db_1_topleft = copyroot.deepcopy(db)

# db_1_topleft.get_y = ColReader(['point_topleft_x', 'point_topleft_y',])

# dls_1_topleft =  db_1_topleft.dataloaders(df)