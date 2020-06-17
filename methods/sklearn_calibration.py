import autograd.numpy as np
import matplotlib.pyplot as plt
import numpy as np

from autograd import grad
from autograd.misc.optimizers import adam, sgd
from collections import Counter
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

def most_overconfident(conf,preds,labels):
    wrong_inds = []
    for i in range(len(labels)):
        if preds[i] != labels[i]:
            wrong_inds.append(i)
    conf_count = Counter()
    for ind in wrong_inds:
        conf_count[ind] = conf[ind]
    return conf_count.most_common()

def calibrate_model(est,method, xval, yval):
    """
    Calibrate model:
        - est: is a pretrained sklearn model
        - method is the type of calibration being performed (eg isotonic or sigmoid)
        - xval is the x-validation data
        - yval is y-validation data
    returns:
        - Calibrated classifier
    """
    calibrated_classifier = CalibratedClassifierCV(est,method,cv='prefit')
    calibrated_classifier.fit(xval,yval)
    return calibrated_classifier

def measure_calibration(est,xtest,ytest,plot_path,probs=None):
    """
    Measure calibration of model:
        - est: an sklearn model
        - xtest,ytest: testing data
        - plot_path: where to save plot
    Return: Brier score
    Saves calibration plot
    """
    fig_index = 1
    name = 'isotonic'
    if probs is not None:
        prob_pos = probs
    else:
        prob_pos = est.predict_proba(xtest)[:, 1]
    clf_score = brier_score_loss(ytest, prob_pos, pos_label=ytest.max())
    fraction_of_positives, mean_predicted_value = calibration_curve(ytest, prob_pos, n_bins=10)
    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s (%1.3f)" % ('iso', clf_score))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
             histtype="step", lw=2)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots: '+plot_path)

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    return clf_score


def _certainty_bounder(y_probs,y_preds,y_labels,thresh=.95):
    prob_increment = .01
    thresh_sat = False 
    cur_prob = 0
    while not thresh_sat:
        assert cur_prob != 1.0
        inds = y_probs > cur_prob
        bin_size = y_preds[inds].shape[0]
        frac_correct = sum(y_preds[inds] == y_labels[inds])/bin_size
        if frac_correct >= thresh:
            return cur_prob
        cur_prob += prob_increment 


def ece_mce_error(y_probs,y_preds,y_labels,bin_range = (0,1),num_bins = 10,plot=False):
    #bins = [((i-1)/num_bins,i/num_bins) for i in range(1,num_bins+1)]
    bins_granularity = (bin_range[1] - bin_range[0])/num_bins
    bins = [((i-1)*bins_granularity+bin_range[0],(i)*bins_granularity+bin_range[0]) for i in range(1,num_bins+1)]
    print(bins)
    print(set(y_probs))
    #bins = [b for b in bins if b[0] >= bin_range[0] and b[1] <= bin_range[1]]
    filter_inds = (y_probs <= bin_range[1])*(y_probs >= bin_range[0])
    #import IPython
    #IPython.embed()
    y_probs = y_probs[filter_inds]
    y_preds = y_preds[filter_inds]
    y_labels = y_labels[filter_inds]
    mce_bin_losses = []
    ece_bin_losses = []
    num_points = y_preds.shape[0]
    point_count = []
    for i,b in enumerate(bins):
        if i == 0:
            inds = (y_probs <= b[1])*(y_probs >= b[0])
        else:
            inds = (y_probs <= b[1])*(y_probs > b[0])
            
          
        if sum(inds) > 0:
            bin_size = y_preds[inds].shape[0]
            frac_correct = sum(y_preds[inds] == y_labels[inds])/bin_size
            expected_correct = sum(y_probs[inds])/bin_size
            mce_bin_losses.append(np.abs(frac_correct-expected_correct))
            point_count.append(sum(inds))
            ece_bin_losses.append(bin_size/num_points*np.abs(frac_correct-expected_correct))
        else:
            ece_bin_losses.append(0)
    if  sum(ece_bin_losses) > .5:
        print('warning ece greater than .5')
    if num_points == 0:
        return 0,0,0
    if len(mce_bin_losses) ==0:
        import IPython
        IPython.embed()

    if plot is not None:
        measure_calibration(None,None,np.array([y_preds[i] == y_labels[i] for i in range(len(y_preds))]),plot,probs=y_probs)
        
        
        
       
    return sum(ece_bin_losses), max(mce_bin_losses), mce_bin_losses, point_count, num_points



def determine_thresh(est,xtest,ytest,thresh=.95):
    preds = est.predict_proba(xtest)
    probs = np.max(preds,axis=1)
    y_pred = est.predict(xtest)
    thresh = _certainty_bounder(preds,probs,y_pred,thresh)
    return thresh 

def measure_ece_mce(est,xtest,ytest,bin_range = (0,1),num_bins=10,plot=False):
    preds = est.predict_proba(xtest)
    probs = np.max(preds,axis=1)
    y_pred = est.predict(xtest)
    ece_error, mce_error, mce_bin_losses, point_count, num_points = ece_mce_error(probs,y_pred,ytest,bin_range,num_bins,plot)
    return ece_error, mce_error, mce_bin_losses, point_count, num_points

def softmax(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def nll_loss(probs,labels):
    #import IPython
    #IPython.embed()


    try:
        losses =0
        for i in range(len(labels)):
            losses += np.log(probs[i,labels[i]])
        return -1*losses/len(labels)
        #return -1*np.mean(np.log(probs[:,labels]))
    except:
        print('in nll loss')
        import IPython
        IPython.embed()

def get_acc_for_thresh(est,x,labels,thresh):
    probs = est.predict_proba(x)
    max_probs = np.max(probs,axis=1)
    preds = est.predict(x)
    sel_preds = preds[max_probs > thresh]
    sel_labels = labels[max_probs > thresh]
    return np.mean(sel_preds==sel_labels)



def roc_calib(est,xval,yval,desired_acc):
    probs = est.predict_proba(xval)
    max_probs = np.max(probs,axis=1)
    preds = est.predict(xval)
    granularity = .001
    i = 0
    cur_acc = 0
    while cur_acc < desired_acc and i < 1:
        sel_preds = preds[max_probs > i]
        sel_labels = yval[max_probs > i]
        cur_acc = np.mean(sel_preds==sel_labels)
        i = i + granularity
    print("Threshold is: "+str(i))
    return i 



def scale_temperature(est,xval,yval):
    probs = est.predict_proba(xval)
    def temp_loss(t,iter):
        t = np.maximum(t,1e-8)
        probs_t = softmax(probs/t)
        loss = nll_loss(probs_t,yval)
        return loss 
    print_freq = 100
    def print_perf(temp,iter,gradient):
        if iter % print_freq == 0:
            print(str(iter)+': '+str(temp)+' '+str(temp_loss(temp,iter)))
    grad_temp = grad(temp_loss)
    tol = .0001
    max_iter = 201
    step_size = .05
    temp_sol = sgd(grad_temp,1.0,step_size=step_size,num_iters=max_iter,callback=print_perf)
    return temp_sol


