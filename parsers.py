import argparse
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--print_metrics', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=float, default=time.time())
    parser.add_argument('--view', type=list, default=['cmlmd', 'cmld', 'cmcd', 'cdcd', 'cmd', 'cd']) 
    parser.add_argument('--view_num', type=int, default=len(parser.parse_args().view))
    parser.add_argument('--loss', type=str, default='info_nce', help='info_nce, bpr, hinge, bce, weighted_bce')
    parser.add_argument('--gnn', type=str, default='gcn', help='gcn, lgcn, sage, gat')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-7)
    parser.add_argument('--threshold', type=int, default=0.6)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--data_path', type=str, default="D:/code/MVHGCN/dataset1/")
    args, _ = parser.parse_known_args()

    return args