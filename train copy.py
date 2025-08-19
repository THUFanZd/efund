import os
import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR

from models import EconomicIndicatorPredictor
from train_func import *

if __name__ == '__main__':
    with open('args.json') as f:
        all_args = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    os.makedirs(f"./res/{args['log_sub_dir']}", exist_ok=True)
    organize_files(f"./res/{args['log_sub_dir']}")
    with open(f'./res/{args["log_sub_dir"]}/args.json', 'w') as f:
        json.dump(args, f, indent=4)

    writer = SummaryWriter(log_dir=f"./logs/{args['log_sub_dir']}")
    ex_arg = {}
    sum_params = 0
    ExTrainDataset = None
    ExTrainDataloader = None
    ExTestDataloader = None
    for i, args in enumerate(all_args):
        # data
        if args.get("data_months", None) == ex_arg.get("data_months", None) and\
            args.get("num_months", None) == ex_arg.get("num_months", None):
            TrainDataset, TrainDataloader, TestDataset, TestDataloader = ExTrainDataset, ExTrainDataloader, None, ExTestDataloader
        else:
            TrainDataset, TrainDataloader, TestDataset, TestDataloader = data_load_split(args)
            ExTrainDataset = TrainDataset
            ExTrainDataloader = TrainDataloader
            ExTestDataloader = TestDataloader
            ex_arg = args

        # model
        if args['model'] == 'lstm':
            model = EconomicIndicatorPredictor(
                merge_input_dim=TrainDataset.get_merge_dim(),
                article_embedding_dim=args['lstm']['article_embedding_dim'],
                macro_dim=args['macro_dim'],
                output_dim=args.get('output_dim', args['macro_dim']),
                merge_lstm_hidden_dim=args['lstm']['merge_hidden_dim'],
                article_lstm_hidden_dim=args['lstm']['article_hidden_dim'],
                monthly_lstm_hidden_dim=args['lstm']['monthly_hidden_dim'],
                dropout_prob=args['lstm']['dropout']
            ).to(device)

        elif args['model'] == 'lstm_notext':
            model = EconomicIndicatorPredictorNoArticle(
                merge_input_dim=TrainDataset.get_merge_dim(),
                macro_dim=args['macro_dim'],
                merge_lstm_hidden_dim=args['lstm']['merge_hidden_dim'],
                monthly_lstm_hidden_dim=args['lstm']['monthly_hidden_dim'],
                dropout_prob=args['lstm']['dropout']
            ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型总参数量为：{total_params}")

        # trainer
        if 'loss' not in args.keys() or args['loss'] == 'mse':
            criterion = nn.MSELoss()
        elif args['loss'] == 'l1':
            criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
        if args['lr_strategy'] == 'expo':
            scheduler = ExponentialLR(optimizer, gamma=args['gamma'])
        elif args['lr_strategy'] == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=args['epoch_num'] * len(TrainDataloader))
        else:
            raise ValueError("Wrong lr strategy parameter. Accept choices: 'cosine', 'expo'")

        sum_params += total_params

        train(model, criterion, optimizer, scheduler, writer, TrainDataloader, TestDataloader, args, device, i)
    
    writer.add_scalar('# of parameters', sum_params, 0)