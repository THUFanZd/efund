import torch
import torch.nn as nn
import torch.nn.functional as F

class TextMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2):
        super(TextMLP, self).__init__()
        self.norm = nn.BatchNorm1d(input_dim)
        self.drop = nn.Dropout(p=dropout)
        
        self.fc_layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                self.fc_layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                self.fc_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))

        self.output = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        # print('after flatten, shape: ', x.shape)
        normed = self.norm(x)
        dropped = self.drop(normed)
        y = dropped

        for fc in self.fc_layers:
            y = F.relu(fc(y))
        
        output = self.output(y)
        return output
    

class TextAvgMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2):
        super(TextAvgMLP, self).__init__()
        self.norm = nn.BatchNorm1d(input_dim)
        self.drop = nn.Dropout(p=dropout)
        
        self.fc_layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                self.fc_layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                self.fc_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))

        self.output = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        # 先对序列维度求平均 [batch_size, seq_len, embedding_dim] -> [batch_size, embedding_dim]
        x = x.mean(dim=1)
        normed = self.norm(x)
        dropped = self.drop(normed)
        y = dropped

        for fc in self.fc_layers:
            y = F.relu(fc(y))
        
        output = self.output(y)
        return output


class TextLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.2):  # 修改参数列表
        super(TextLSTM, self).__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.drop = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)  # 修改输入维度
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        normed = self.norm(x)  # [batch, seq_len, input_dim]
        if len(normed.size()) == 2:
            normed = normed.unsqueeze(1)
        dropped = self.drop(normed)
        output, (hidden, cell) = self.lstm(dropped)

        return self.fc(hidden[-1])



class TextCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, output_dim, dropout=0.2):
        super(TextCNN, self).__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.drop = nn.Dropout(p=dropout)
        
        # 使用不同尺度的卷积核并行工作
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=dim, kernel_size=k),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)  # 全局最大池化
            )
            for dim, k in zip(out_channels, kernel_sizes)
            #          输出维度即卷积核个数  三种不同kernel尺寸
            # 注意，一个卷积核就是128通道的，所以这个网络实际上，对于一个尺寸的卷积核来说，只进行了一次扫描
            # 而不是说设置了128的out_channels就会有128个卷积核，还是只有一个，只不过卷积是在所有维度上同时进行的
            # 不对，卷积核是300通道的啊，卷完应该只有一个数，所以128个通道就说明有128个核
        ])
        
        self.output = nn.Linear(sum(out_channels), output_dim)

    def forward(self, x):
        # 输入形状: [batch, 16, 300]
        x = self.norm(x)
        x = self.drop(x)
        
        # 转换维度为 [batch, 300, 16]
        x = x.permute(0, 2, 1)
        
        # 并行卷积处理
        features = [branch(x).squeeze(-1) for branch in self.conv_branches]
        concatenated = torch.cat(features, dim=1)
        return self.output(concatenated)



class TextTransformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers=2, num_heads=4, dropout=0.2):
        super(TextTransformerDecoder, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, input_dim))
        self.norm = nn.LayerNorm(input_dim)
        self.drop = nn.Dropout(p=dropout)
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))  # 可学习的查询向量
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 输入形状: [batch_size, seq_len, input_dim]
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]  # 添加位置编码
        x = self.norm(x)
        x = self.drop(x)
        
        # 扩展查询向量匹配batch大小
        query = self.query.expand(x.size(0), -1, -1)
        
        # 通过Transformer解码器
        x = self.decoder(tgt=query, memory=x)
        
        # 取最后一个查询位置的结果
        return self.fc(x[:, -1, :])




class EconomicIndicatorPredictor(nn.Module):
    def __init__(self, 
                 merge_input_dim,
                 article_embedding_dim,
                 macro_dim,
                 merge_lstm_hidden_dim=64,
                 article_lstm_hidden_dim=64,
                 monthly_lstm_hidden_dim=128,
                 dropout_prob=0.3):
        super(EconomicIndicatorPredictor, self).__init__()

        self.dropout_prob = dropout_prob

        # 金融指数每日LSTM → 每月一个向量
        self.financial_lstm = nn.LSTM(input_size=merge_input_dim, 
                                      hidden_size=merge_lstm_hidden_dim,
                                      batch_first=True)

        # 文章LSTM → 每月一个向量
        self.article_lstm = nn.LSTM(input_size=article_embedding_dim,
                                    hidden_size=article_lstm_hidden_dim,
                                    batch_first=True)

        # Dropout after LSTM outputs
        self.dropout_financial = nn.Dropout(p=dropout_prob)
        self.dropout_article = nn.Dropout(p=dropout_prob)

        # 月度输入维度：金融向量 + 文章向量 + 宏观指标向量
        monthly_input_dim = merge_lstm_hidden_dim + article_lstm_hidden_dim + macro_dim

        # 月度LSTM
        self.monthly_lstm = nn.LSTM(input_size=monthly_input_dim,
                                    hidden_size=monthly_lstm_hidden_dim,
                                    batch_first=True)

        self.dropout_monthly = nn.Dropout(p=dropout_prob)

        # 输出 MLP 头
        self.mlp_head = nn.Sequential(
            nn.Linear(monthly_lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, macro_dim)
        )

    def forward(self, financial_seq, article_seq, macro_seq):
        batch_size, n_months, days_per_month, _ = financial_seq.shape

        # Step 1: encode financial index
        financial_seq = financial_seq.view(-1, days_per_month, financial_seq.size(-1))  # (B * n, D, a)
        _, (financial_hidden, _) = self.financial_lstm(financial_seq)
        financial_monthly = financial_hidden[-1].view(batch_size, n_months, -1)
        financial_monthly = self.dropout_financial(financial_monthly)

        # Step 2: encode article
        article_seq = article_seq.view(-1, article_seq.size(2), article_seq.size(3))   # (B * n, L, D)
        _, (article_hidden, _) = self.article_lstm(article_seq)
        article_monthly = article_hidden[-1].view(batch_size, n_months, -1)
        article_monthly = self.dropout_article(article_monthly)

        # Step 3: concat features
        monthly_features = torch.cat([financial_monthly, article_monthly, macro_seq], dim=-1)

        # Step 4: temporal modeling over months
        _, (monthly_hidden, _) = self.monthly_lstm(monthly_features)
        monthly_hidden_last = self.dropout_monthly(monthly_hidden[-1])  # (B, H)

        # Step 5: output
        output = self.mlp_head(monthly_hidden_last)
        return output


class EconomicIndicatorPredictorNoArticle(nn.Module):
    def __init__(self, 
                 merge_input_dim,
                 macro_dim,
                 merge_lstm_hidden_dim=64,
                 monthly_lstm_hidden_dim=128,
                 dropout_prob=0.3):
        super(EconomicIndicatorPredictorNoArticle, self).__init__()

        self.dropout_prob = dropout_prob

        # 金融指数每日LSTM → 每月一个向量
        self.financial_lstm = nn.LSTM(input_size=merge_input_dim, 
                                      hidden_size=merge_lstm_hidden_dim,
                                      batch_first=True)

        # Dropout after LSTM outputs
        self.dropout_financial = nn.Dropout(p=dropout_prob)

        # 月度输入维度：金融向量 + 宏观指标向量
        monthly_input_dim = merge_lstm_hidden_dim + macro_dim

        # 月度LSTM
        self.monthly_lstm = nn.LSTM(input_size=monthly_input_dim,
                                    hidden_size=monthly_lstm_hidden_dim,
                                    batch_first=True)

        self.dropout_monthly = nn.Dropout(p=dropout_prob)

        # 输出 MLP 头
        self.mlp_head = nn.Sequential(
            nn.Linear(monthly_lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, macro_dim)
        )

    def forward(self, financial_seq, article_seq, macro_seq):
        batch_size, n_months, days_per_month, _ = financial_seq.shape

        # Step 1: encode financial index
        financial_seq = financial_seq.view(-1, days_per_month, financial_seq.size(-1))  # (B * n, D, a)
        _, (financial_hidden, _) = self.financial_lstm(financial_seq)
        financial_monthly = financial_hidden[-1].view(batch_size, n_months, -1)
        financial_monthly = self.dropout_financial(financial_monthly)

        # Step 3: concat features (without article features)
        monthly_features = torch.cat([financial_monthly, macro_seq], dim=-1)

        # Step 4: temporal modeling over months
        _, (monthly_hidden, _) = self.monthly_lstm(monthly_features)
        monthly_hidden_last = self.dropout_monthly(monthly_hidden[-1])  # (B, H)

        # Step 5: output
        output = self.mlp_head(monthly_hidden_last)
        return output