import torch
import os
import pandas as pd
import numpy as np
from test_cache import target_date
from models import EconomicIndicatorPredictor
from train_func import data_load_split


# 加载测试args
with open('./test_args.json', 'r') as f:
    import json
    test_args = json.load(f)
test_model_name = test_args['test_model_name']

with open(f'./res/{test_model_name}/args.json', 'r') as f:
    import json
    all_args = json.load(f)

log_sub_dir = all_args[0]["log_sub_dir"]
output_dim = all_args[0].get('output_dim', all_args[0]['macro_dim'])
models = []
for i, args in enumerate(all_args):
    args["log_sub_dir"] = log_sub_dir
    train_cache_path = f'./cache/{i}'
    TrainDataset, TrainDataloader, TestDataset, TestDataloader = data_load_split(args, train_cache_path)
    macro_dim = args['macro_dim'] if not args.get('add_month', False) else args["macro_dim"] + 1
    model = EconomicIndicatorPredictor(
        merge_input_dim=TrainDataset.get_merge_dim(),
        article_embedding_dim=args['lstm']['article_embedding_dim'],
        macro_dim=macro_dim,
        output_dim=len(args['group_indices']),
        merge_lstm_hidden_dim=args['lstm']['merge_hidden_dim'],
        article_lstm_hidden_dim=args['lstm']['article_hidden_dim'],
        monthly_lstm_hidden_dim=args['lstm']['monthly_hidden_dim'],
        dropout_prob=args['lstm']['dropout']
    )

    # 加载模型参数
    checkpoint_path = f"./res/{args['log_sub_dir']}/model_{i}_epoch_{args['epoch_num']}.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    models.append(model)


# 存储结果
records = []
macro_dim = all_args[0]['macro_dim']

for i, date in enumerate(target_date):
    try:
        preds = []
        for j, model in enumerate(models):
            macro_x, merged_x, conf_x, y = torch.load(os.path.join(f'./cache_test/{j}', f'{i}.pt'))
            merged_x = merged_x.unsqueeze(0)
            conf_x = conf_x.unsqueeze(0)
            macro_x = macro_x.unsqueeze(0)

            with torch.no_grad():
                single_pred = model(merged_x, conf_x, macro_x).squeeze(0)  # shape: [output_dim]
                preds.append(single_pred)

        pred = torch.zeros(macro_dim)
        for j in range(len(models)):
            pred[all_args[j]['group_indices']] = preds[j]

        pred_np = pred.numpy()
        y_np = y.numpy()

        row = {'date': date.strftime('%Y-%m-%d')}
        for j in range(output_dim):
            row[f'pred_{j}'] = pred_np[j]
            row[f'true_{j}'] = y_np[j]
            row[f'mae_{j}'] = abs(pred_np[j] - y_np[j])
            row[f'dir_correct_{j}'] = int(np.sign(pred_np[j]) == np.sign(y_np[j]))

        records.append(row)
        print(f"{i}: {row['date']} processed.")

    except Exception as e:
        print(f"Sample {i} failed: {e}")

# 保存为 CSV
df = pd.DataFrame(records)
os.makedirs('./test_res', exist_ok=True)
df.to_csv(f'./test_res/test_results_{test_model_name}.csv', index=False)
print("Saved detailed results to test_results.csv")
