import datetime
import gc
import random
import numpy as np
import torch
import torch.optim as optim
from model import Model
from utils import get_metrics, output_result, stop
from dataload import dataload
from parsers import get_args

def main(args):
    starttime0 = datetime.datetime.now()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(int(args.seed))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    As, feature,pos,neg,ncirc,ndis = dataload(args)

    metrics = get_metrics()
    res = metrics.init()

    endtime0 = datetime.datetime.now()
    time = (endtime0 - starttime0).seconds
    print(f"Loading data time: {time}s")

    for i in range(5): 
        model = Model(device, args.view_num, args.gnn, ncirc, ndis, args.hidden_dim, args.dropout)
        model.to(device)
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        fold = [0, 1, 2, 3, 4]
        fold.remove(i)

        res = metrics.init2(res)

        for epoch in range(args.epochs):
            starttime1 = datetime.datetime.now()
            gc.collect()
            torch.cuda.empty_cache()
            model.train()
            res = metrics.init3(res)

            for j in fold: 
                score= model.forward(As[j], feature[j], i)
                loss = model.loss(args.loss, score,pos[j],neg[j],args.temperature)

                loss = loss.requires_grad_()
                opt.zero_grad()
                loss.backward()
                opt.step()


              
            res  = metrics.test(model, As[i], feature[i] ,res, i, pos[i],neg[i],ncirc, args.threshold)

            endtime1 = datetime.datetime.now()
            time = (endtime1 - starttime1).seconds


            if args.print_metrics:
                print("fold: {}, time: {}s Epoch: {} Loss: {:.2f} AUC: {:.4f},{:.4f}"  \
                    .format(i+1, time, epoch, loss, res['roc_auc'][i], res['auc'].mean()))
            
            if stop(res, args, i, epoch):
                break


    output_result(res)
    endtime2 = datetime.datetime.now()
    time = (endtime2 - starttime0).seconds      
    print(f"Total time: {time}s")

if __name__ == '__main__':
    args = get_args()
    main(args)








