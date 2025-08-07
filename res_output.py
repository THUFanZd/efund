import torch
import os
import pandas as pd
import numpy as np
from test_cache import cache_path, target_date
from models import EconomicIndicatorPredictor
from train_func import data_load_split


# 加载 args
with open('./test_args.json', 'r') as f:
    import json
    test_args = json.load(f)
test_model_name = test_args['test_model_name']
eval_epoch = test_args['eval_epoch']

with open(f'./res/{test_model_name}/args.json', 'r') as f:
    import json
    args = json.load(f)

output_dim = args.get('output_dim', args['macro_dim'])

TrainDataset, TrainDataloader, TestDataset, TestDataloader = data_load_split(args)
# 初始化模型
model = EconomicIndicatorPredictor(
    merge_input_dim=TrainDataset.get_merge_dim(),
    article_embedding_dim=args['lstm']['article_embedding_dim'],
    macro_dim=args['macro_dim'],
    output_dim=output_dim,
    merge_lstm_hidden_dim=args['lstm']['merge_hidden_dim'],
    article_lstm_hidden_dim=args['lstm']['article_hidden_dim'],
    monthly_lstm_hidden_dim=args['lstm']['monthly_hidden_dim'],
    dropout_prob=args['lstm']['dropout']
)

# 加载模型参数
checkpoint_path = f"./res/{args['log_sub_dir']}/epoch_{eval_epoch}.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 存储结果
records = []

for i, date in enumerate(target_date):
    try:
        macro_x, merged_x, conf_x, y = torch.load(os.path.join(cache_path, f'{i}.pt'))

        merged_x = merged_x.unsqueeze(0)
        conf_x = conf_x.unsqueeze(0)
        macro_x = macro_x.unsqueeze(0)

        with torch.no_grad():
            pred = model(merged_x, conf_x, macro_x).squeeze(0)  # shape: [output_dim]

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
