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
        args = json.load(f)
        assert type(args) == dict
        print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    os.makedirs(f"./res/{args['log_sub_dir']}", exist_ok=True)
    organize_files(f"./res/{args['log_sub_dir']}")
    with open(f'./res/{args["log_sub_dir"]}/args.json', 'w') as f:
        json.dump(args, f, indent=4)

    # data
    TrainDataset, TrainDataloader, TestDataset, TestDataloader = data_load_split(args)

    # model
    if args['model'] == 'lstm':
        model = EconomicIndicatorPredictor(
            merge_input_dim=TrainDataset.get_merge_dim(),
            article_embedding_dim=args['lstm']['article_embedding_dim'],
            macro_dim=args['macro_dim'],
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

    # run
    organize_files(f"./logs/{args['log_sub_dir']}")
    writer = SummaryWriter(log_dir=f"./logs/{args['log_sub_dir']}")
    writer.add_scalar('# of parameters', total_params, 0)

    train(model, criterion, optimizer, scheduler, writer, TrainDataloader, TestDataloader, args, device)
    test(model, TestDataloader, writer, device)
