import os, sys
import copy as copyroot
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt

from fastai.basics import *    
from fastai.vision.all import *

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def point_from_img(fn, path):

    img_np = np.array(Image.open(str(path) + fn))

    h, w   =          img_np.shape

    img_bool =         np.where(img_np > 0, 1, 0)
    
    row_sums =         img_bool.argmax(axis=1)
    col_sums =         img_bool.argmax(axis=0)
    
    binary_row_sums =  np.where(row_sums > 0, 1, 0)
    binary_col_sums =  np.where(col_sums > 0, 1, 0)

    top_row_index =    binary_row_sums.tolist().index(1)
    left_col_index =   binary_col_sums.tolist().index(1)

    bottom_row_index = h - binary_row_sums.tolist()[::-1].index(1)
    right_col_index =  w - binary_col_sums.tolist()[::-1].index(1)

    top_row_values =   img_np[top_row_index,:]
    topleftmost_index = (top_row_values > 0).tolist().index(True)

    center_row_index = int((top_row_index + bottom_row_index) / 2)
    center_col_index = int((left_col_index + right_col_index) / 2)

    pixel_sum =        img_np.sum().sum()

    ret = {
        'scalar_top':       top_row_index,
        'scalar_bottom':    bottom_row_index,
        'scalar_pxsum':     pixel_sum,
        'point_topleft_x':  topleftmost_index,
        'point_topleft_y':  top_row_index, 
        'point_center_x':   center_row_index, 
        'point_center_y':   center_col_index,
    }

    return ret


def build_df(path):
    '''
        return df for MNIST_TINY with features:
            fn          
            digit_class 
            scalar_top
            scalar_bottom
            scalar_pxsum
            point_topleft_x
            point_topleft_y
            point_center_x
            point_center_y
    '''

    # load basic info  --------------------------------

    df = pd.DataFrame({'fn':[],'digit_class':[]})

    for digit_cat in os.listdir(path/'train'):
        
        tmp = [f'/train/{digit_cat}/{e}' for e in 
                os.listdir(path/'train'/str(digit_cat))] 
        
        df_tmp = pd.DataFrame({'fn': tmp})
        
        df_tmp['digit_class'] = digit_cat
        
        df = pd.concat((df, df_tmp))
        
    df['digit_class'] = df['digit_class'].astype('int')
    
    df.reset_index(drop=True, inplace=True)

    # augment features with function --------------------
    
    features = [
        'scalar_top',
        'scalar_bottom',
        'scalar_pxsum',
        'point_topleft_x',
        'point_topleft_y',
        'point_center_x',
        'point_center_y',
    ]

    def unpack_vals(dict_ret):
        ret = []
        for f in features:
            ret.append(dict_ret[f])
        return ret

    vals = [point_from_img(fn, path) for fn in df['fn']]

    df_feats = pd.DataFrame(vals, columns=features)
    
    df = pd.concat((df,df_feats), axis=1)

    # convert to float?
    # for col in df.columns:
    #     if col not in non_float_cols:

    return df

def eda_fig_1(df, path):

    features = [
            'scalar_top',
            'scalar_bottom',
            'scalar_pxsum',
            'point_topleft_x',
            'point_topleft_y',
            'point_center_x',
            'point_center_y',
        ]

    db       =   DataBlock(blocks=(ImageBlock(cls=PILImageBW), 
                                RegressionBlock(n_out=len(features))), 
                        splitter=RandomSplitter(),
                        get_x=ColReader('fn', pref=path),
                        get_y=ColReader(features),
                        )

    dls = db.dataloaders(df)

    d_features = {e:i for i,e in enumerate(features)}
    d_features['scalar_bottom']

    b = dls.one_batch()

    b_decoded = dls.decode_batch(b, max_n=64)

    i = 0
    (b        [1][i][d_features['scalar_pxsum']],
    b_decoded[i][1][d_features['scalar_pxsum']])

    b =         dls.one_batch()
    b_decoded = dls.decode_batch(b, max_n=64)

    rows, cols = 2,5
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(3*cols,3*rows + 2))
    axr = ax.ravel()

    for i in range(rows*cols):
        
        yi = b_decoded[i][1]

        b_decoded[i][0].show(ctx=axr[i]);

        x  = yi[d_features['point_topleft_x']]
        y  = yi[d_features['point_topleft_y']]
        TensorPoint((x,y)).show(ctx=axr[i], s=300, marker='o', c='red')

        x  = yi[d_features['point_center_x']]
        y  = yi[d_features['point_center_y']]
        TensorPoint((x,y)).show(ctx=axr[i], s=300, marker='o', c='green')

        scalar_pxsum = yi[d_features['scalar_pxsum']]
        scalar_top = yi[d_features['scalar_top']]
        scalar_bottom = yi[d_features['scalar_bottom']]
        tile_title = f'pxsum: {scalar_pxsum}\ntop: {scalar_top}\nbottom: {scalar_bottom}'
        axr[i].set_title(tile_title);
        axr[i].axis('on')
    fig_title = (
    '''
    Display items with corresponding Y's:
    Scalar targets are printed above the items
    Point targets are shown on image: (Red: topleft, Green: center)
    ''')
    # fig.suptitle(fig_title);
    print(fig_title)
    plt.show()

def img_pt_plot(
        ys,
        legend = None,
        b_flip_y=True,
        colors = ['blue', 'green', 'red'],
        figsize=(4,4),
        ):

    plt.figure(figsize=figsize)
    
    flip_factor = -1 if b_flip_y else 1

    for i,y in enumerate(ys):
        plt.scatter(y[:,0], y[:,1].mul(flip_factor), 
                    color=colors[i], alpha=0.4);
    
    plt.ylim(-1,1); plt.xlim(-1,1);
    
    if b_flip_y:
        plt.ylabel('coords reversed on y for display')
        plt.xlabel('coords same for x')
    
    if legend is not None:
        plt.legend(['topleft', 'center']);
    
    plt.title('position of target points in dataset  \n (flow field position)');


def train_history_dualplot(
        train_history,
        fig2_startx,
        baseline_err = None,
        ):

    fig, ax = plt.subplots(1,2,figsize=(10,5))
    legend = ['train','valid']
    if baseline_err is not None: legend += ['baseline err']

    ax[0].plot(train_history['train_loss'])
    ax[0].plot(train_history['valid_loss'])
    if baseline_err is not None:
        ax[0].hlines(baseline_err, *ax[0].get_xlim(), linestyle='--')
    ax[0].legend(legend);
    ax[0].set_xlabel('all epochs'); ax[0].set_ylabel('MSE')

    ax[1].plot(train_history['train_loss'][fig2_startx:])
    ax[1].plot(train_history['valid_loss'][fig2_startx:]);
    if baseline_err is not None:
        ax[1].hlines(baseline_err, *ax[1].get_xlim(), linestyle='--')
    ax[1].legend(legend);
    ax[1].set_xlabel('late epochs');

def build_dls(target='topleft'):

    path = untar_data(URLs.MNIST_TINY)
    df = build_df(path)
    df.head(2)

    y_names = [
        'point_topleft_x', 
        'point_topleft_y',
        'point_center_x',
        'point_center_y'
        ]

    db =   DataBlock(blocks=(ImageBlock(cls=PILImageBW), 
                            PointBlock), 
                    splitter=RandomSplitter(seed=0),
                    get_x=ColReader('fn', pref=path),
                    )

    db_1_topleft = copyroot.deepcopy(db)
    db_1_center  = copyroot.deepcopy(db)
    db_2         = copyroot.deepcopy(db)    

    def set_get_y(db, cr):
        db.get_y = cr
        db.getters[db.n_inp:] = L(db.get_y)

    set_get_y( db_1_topleft, ColReader(y_names[:2]) )
    set_get_y( db_1_center,  ColReader(y_names[2:]) )
    set_get_y( db_2,         ColReader(y_names) )

    dl_1_topleft = db_1_topleft.dataloaders(df)
    dl_1_center  = db_1_center.dataloaders(df)
    dl_2         = db_2.dataloaders(df)

    if   target == 'topleft': return dl_1_topleft
    elif target == 'center':  return dl_1_center
    elif target == 'dual':    return dl_2
    else: raise Exception('f{target} not a valid `target`')

def build_tabular_df(path):
    
    path = untar_data(URLs.MNIST_TINY)
    df = build_df(path)

    # df2 - pixel values
    df2 = pd.DataFrame([])
    for i, row in df.iterrows():    
        img_np = np.array(Image.open(str(path) + row['fn']))
        df2 = pd.concat((df2, pd.Series(img_np.flatten())), axis=1)
    df2 = df2.T
    df2.reset_index(inplace=True, drop=True)

    # df3 - pts features

    feats = ['pts11','pts12','pts22']
    featsnet = FeatsNet(feats = feats)

    # 2*(4 + 16 + 16)
    cols =  []
    cols += [f'pts11_{i}' for i in range(8)]
    cols += [f'pts12_{i}' for i in range(32)]
    cols += [f'pts22_{i}' for i in range(32)]

    df3 = pd.DataFrame([])
    for i, row in df.iterrows():    
        img_np = np.array(Image.open(str(path) + row['fn']))
        tmp = featsnet.build_feats(
                    torch.tensor(img_np).unsqueeze(0)
                    ).squeeze(0).tolist()
        df3 = pd.concat((df3, pd.Series(tmp)), axis=1)
    df3 = df3.T
    df3.reset_index(inplace=True, drop=True)
    df3.columns = cols

    y_names = ['scalar_pxsum', 'point_topleft_x', 'point_topleft_y',
            'point_center_x', 'point_center_y']

    df2_y = df[y_names]
    df2 = pd.concat((df2, df3, df2_y), axis=1)

    y_cols = y_names
    x_cols = [col for col in df2.columns if col not in y_cols]

    X = df2[x_cols]
    Y = df2[y_cols]

    return X, Y
