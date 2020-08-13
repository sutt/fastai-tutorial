from matplotlib import pyplot as plt
from IPython.display import display
from fastai2.vision.all import *


def show_figures(dls, preds):
    
    # establish actuals --------
    yhat, yactual = preds[0], preds[1]

    y = tensor([])
    counter = 0
    for _x, _y in dls.train:
        y = torch.cat((y,_y))
        counter +=1
        if counter == 10: 
            break

    # build mean point --------
    xbar, ybar = yactual.view(-1,2).mean(dim=0)

    xbar_train, ybar_train = y.view(-1,2).mean(dim=0)

    # build tables ------------
    df_center = pd.DataFrame(
                {'x-mean':[xbar_train.tolist(), xbar.tolist()], 
                 'y-mean':[ybar_train.tolist(), ybar.tolist()]},
                 index=['training', 'validation'])

    mse_train_train = mse(
                    tensor([[xbar_train, ybar_train] 
                           for _ in range(len(y))])
                    , y)

    mse_valid_train = mse(
                        tensor([[xbar_train, ybar_train] 
                                for _ in range(len(yactual))])
                        , yactual)

    mse_valid_valid = mse(
                        tensor([[xbar, ybar] 
                            for _ in range(len(yactual))])
                        , yactual)

    mse_valid_model = mse(yhat, yactual)

    df_err = pd.DataFrame(
            {'Actual': ['Train', 'Valid', 'Valid', 'Valid'],
             'Predicted': ['mean-point: train', 'mean-point: train', 'mean-point: valid', 'model'],
             'MSE':[
                     mse_train_train.tolist(),
                     mse_valid_train.tolist(),
                     mse_valid_valid.tolist(),
                     mse_valid_model.tolist(),
             ]
            })

    df_err['RMSE'] = df_err['MSE'] ** 0.5

    # display tables -------------
    print('\nDifferent Mean-Points for Training & Validation:')
    display(df_center.round(3))
    print('\nMSE on diff predictions types:')
    display(df_err.round(4))

    # figure 1 --------------------

    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].scatter(y.view(-1,2)[:,0], y.view(-1,2)[:,1], alpha=0.4)
    ax[0].set_ylim(-1,1)
    ax[0].set_xlim(-1,1);

    ax[0].scatter(yactual.view(-1,2)[:,0], yactual.view(-1,2)[:,1], 
                alpha=0.3, color='green')
    ax[0].scatter(xbar_train, ybar_train, c='blue', marker='x');
    ax[0].scatter(xbar, ybar, c='black', marker='x');

    ax[0].legend(['y train', 'y validation', 
                'mean of y (in train)', 'mean of y (in valid)'])
    ax[0].set_title('Y-Actual: Train vs. Valid \n (sample: 10 batches of 64)');
    ax[0].set_xlabel('full image field')

    ax[1].scatter(yactual.view(-1,2)[:,0], yactual.view(-1,2)[:,1], c='b', alpha=0.4)
    ax[1].scatter(yhat.view(-1,2)[:,0], yhat.view(-1,2)[:,1], c='g', alpha=0.4)
    ax[1].scatter(xbar, ybar, c='blue', marker='x', s=100)
    ax[1].scatter(xbar_train, ybar_train, c='red', marker='x', s=100)
    ax[1].legend(['actual', 'predicted', 'mean, valid', 'mean, train'])
    ax[1].set_title('Actual and Model-Predicted coords \n Validation Set');
    ax[1].set_xlabel('clipped image field');

    plt.show()

    # figure 2 ----------------------

    pad = 1.1
    min_val = yactual.view(-1,2).min(dim=0).values.min() * pad
    max_val = yactual.view(-1,2).max(dim=0).values.max() * pad
    ll = np.linspace(min_val, max_val);
    min_val, max_val

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    ax[0].scatter(yactual[:,0,0],  yhat[:,0] )
    ax[0].plot(ll, ll, linestyle='--' )
    ax[0].set_title('X coord')
    ax[0].set_xlabel('actual')
    ax[0].set_ylabel('predicted')
    ax[1].scatter(yactual[:,0,1],  yhat[:,1] )
    ax[1].plot(ll, ll, linestyle='--' )
    ax[1].set_title('Y coord')
    ax[1].set_xlabel('actual')
    fig.suptitle('Validation: Predicted vs Actual');

    plt.show()

    # calculate residuals ---------------

    mean_resids  = tensor([xbar, ybar]) - yactual
    model_resids = yhat - yactual.view(-1,2)

    dist_mean_resid = ((mean_resids.view(-1,2)**2)
                    .sum(dim=1)**0.5)
    dist_model_resid = ((model_resids.view(-1,2)**2)
                        .sum(dim=1)**0.5)

    avg_dist_mean_resid  = dist_mean_resid.mean()
    avg_dist_model_resid = dist_model_resid.mean()

    avg_dist_mean_resid, avg_dist_model_resid

    # figure 3 --------------------------

    pad = 1.1
    min_val = mean_resids.view(-1,2).min(dim=0).values.min() * pad
    max_val = mean_resids.view(-1,2).max(dim=0).values.max() * pad
    min_val, max_val

    fig, ax = plt.subplots(1,2,figsize=(10,5))

    ax[0].scatter(mean_resids.view(-1,2)[:,0], 
                mean_resids.view(-1,2)[:,1],
                c='blue', alpha=0.4)

    ax[0].scatter(model_resids.view(-1,2)[:,0], 
                model_resids.view(-1,2)[:,1],
                c='green', alpha=0.4)

    circle_mean =  plt.Circle((0,0), avg_dist_mean_resid, 
                            fill=False, linestyle='--', color='b')
    circle_model = plt.Circle((0,0), avg_dist_model_resid, 
                            fill=False, linestyle='--', color='g')

    ax[0].add_artist(circle_mean)
    ax[0].add_artist(circle_model)

    ax[0].set_xlim(min_val, max_val)
    ax[0].set_ylim(min_val, max_val);

    ax[0].plot([1e-5, 1e-6], [1e-5, 1e-6], c='blue', linestyle='--')
    ax[0].plot([1e-5, 1e-6], [1e-5, 1e-6], c='green', linestyle='--')

    ax[0].legend([
                'avg mean-resid dist', 'avg model-resid dist',
                'mean residuals', 'model residuals', ])


    ax[0].hlines(0,min_val, max_val, linestyle='--')
    ax[0].vlines(0,min_val, max_val, linestyle='--')

    ax[0].set_title('Residuals Coords, mean vs model \n (predicted - actual)');

    ax[1].hist([ dist_mean_resid.tolist(), 
           dist_model_resid.tolist()
         ], color=['blue','green'], bins=14,
        );
    ax[1].legend(['valid-mean', 'model']);
    ax[1].set_title('Dist Err of Valid-Mean vs. Model');

    plt.show()