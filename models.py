import torch
import torch.nn as nn
import torch.nn.functional as F

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