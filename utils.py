import random
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import h5py
import os
import torch.nn.functional as F


def output_result(res):
    print("AUC: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}" \
    .format(
    res['roc_auc'][0], res['roc_auc'][1], res['roc_auc'][2], res['roc_auc'][3], res['roc_auc'][4], res['roc_auc'].mean()))

def link_prediction(model, adj_val, fea_val, fold, pos_dict, neg_dict, ncirc, th=0.85):
    true_list = []  
    pred_score = []  
    pred_pos_list = []  

    model.eval()
    with torch.no_grad:
        score_ = model.forward(adj_val, fea_val, fold) 
    score_ = score_.detach().cpu().numpy()
    score =  1.0 / (1.0 + np.exp(-score_))
    
    for i in range(ncirc):    
        pos = score[i, pos_dict[i]]
        neg = score[i, neg_dict[i]]

        for j in range(len(pos)):
            true_list.append(1)
            pred_score.append(pos[j])
            if pos[j] > th:
                pred_pos_list.append(1)
            else:
                pred_pos_list.append(0)

        for j in range(len(neg)):
            true_list.append(0)
            pred_score.append(neg[j])
            if neg[j] < th:
                pred_pos_list.append(0)
            else:
                pred_pos_list.append(1)

    y_true = np.array(true_list)
    y_scores = np.array(pred_score)
    y_pred = np.array(pred_pos_list)


    return y_true, y_scores, y_pred

class get_metrics(object):
    def __init__(self):
        pass

    def init(self):
        result = {'roc_auc': np.zeros(5), 'true': [],'scores': [], 'pred': [] , 'auc': np.zeros(5), 
                  'true2': np.empty(5, dtype=object),'scores2': np.empty(5, dtype=object), 'pred2': np.empty(5, dtype=object)
                  }
        
        return result
    
    def init2(self, res):
        res['auc'] = np.zeros(5)
        res['true2'] = np.empty(5, dtype=object)
        res['scores2'] = np.empty(5, dtype=object)
        res['pred2'] = np.empty(5, dtype=object)
        
        return res
    
    def init3(self, res):
        res['true'] = []
        res['scores'] = []
        res['pred'] = []
        
        return res


    def test(self, model, adj_val, fea_val, result, fold, pos_dict, neg_dict, ncirc, th=0.85):    
        true, scores, pred = link_prediction(model, adj_val, fea_val, fold, pos_dict, neg_dict, ncirc, th)
        roc_auc = roc_auc_score(true, scores) 
    
        result['true'].extend(true)
        result['scores'].extend(scores)
        result['pred'].extend(pred)
        result['roc_auc'][fold] = roc_auc
        
        return result

def stop(result, args, fold, epoch):

    if ((result['auc'].min() + result['auc'].mean())/2 > result['roc_auc'][fold] and epoch > args.patience) or epoch == args.epochs-1 : 
        idx = result['auc'].argmax()

        if not os.path.exists(args.data_path + 'results/' + f'{args.view}/'):
            os.mkdir(args.data_path + 'results/' + f'{args.view}/')
        with h5py.File(args.data_path + 'results/' + f'{args.view}/{fold}_{epoch}tsp{args.gnn}{args.hidden_dim}.h5', 'w') as f:
            f.create_dataset('true', data=result['true2'][idx])
            f.create_dataset('scores', data=result['scores2'][idx])
            f.create_dataset('pred', data=result['pred2'][idx])
        
        result['roc_auc'][fold] = result['auc'][idx]

        return True
    else:
        idx = epoch % 5
        result['auc'][idx] = result['roc_auc'][fold]
        result['true2'][idx] = result['true']
        result['scores2'][idx] = result['scores']
        result['pred2'][idx] = result['pred']

        return False

    

    
