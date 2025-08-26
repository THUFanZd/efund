import math
import torch
import torch.nn as nn

class EconomicIndicatorPredictor(nn.Module):
    def __init__(self, 
                 merge_input_dim,
                 article_embedding_dim,
                 macro_dim,
                 output_dim,
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
            nn.Linear(64, output_dim)
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
                 output_dim,
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
            nn.Linear(64, output_dim)
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
    

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding.
    Expects input of shape (S, B, D). Adds position encodings of length S.
    """
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (S, B, D)
        S = x.size(0)
        x = x + self.pe[:S]
        return self.dropout(x)


class TransformerPooler(nn.Module):
    """Helper that builds a TransformerEncoder and returns the representation of a [CLS] token.

    Input is expected in (B, T, Din). If Din != d_model, it will project.
    We prepend a learnable [CLS] token, add PE, pass through encoder, and return the CLS embedding.
    Optionally returns all-time-step embeddings when return_sequence=True.
    """
    def __init__(self,
                 in_dim: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 return_sequence: bool = False):
        super().__init__()
        self.proj = nn.Identity() if in_dim == d_model else nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation='gelu', batch_first=False, layer_norm_eps=layer_norm_eps
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pe = PositionalEncoding(d_model, dropout)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.return_sequence = return_sequence
        self.out_dim = d_model

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        # x: (B, T, Din); key_padding_mask: (B, T), True for padding positions
        B, T, _ = x.shape
        x = self.proj(x)

        # prepend CLS
        cls = self.cls.expand(-1, B, -1)  # (1, B, D)
        x = x.transpose(0, 1)  # (T, B, D)
        x = torch.cat([cls, x], dim=0)  # (T+1, B, D)

        # build mask for transformer (True means masked)
        if key_padding_mask is not None:
            # add False column for CLS (CLS is never masked)
            cls_pad = torch.zeros((B, 1), dtype=torch.bool, device=key_padding_mask.device)
            key_padding_mask = torch.cat([cls_pad, key_padding_mask], dim=1)  # (B, T+1)

        x = self.pe(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (T+1, B, D)

        cls_out = x[0]  # (B, D)
        if self.return_sequence:
            return cls_out, x[1:].transpose(0, 1)  # (B, D), (B, T, D)
        return cls_out


# -----------------------------
# Models: Transformer replacements for LSTM versions
# -----------------------------
class EconomicIndicatorPredictorTransformer(nn.Module):
    """
    Transformer-based drop-in replacement for EconomicIndicatorPredictor.

    Inputs:
      financial_seq: (B, n_months, days_per_month, merge_input_dim)
      article_seq:   (B, n_months, L, article_embedding_dim)
      macro_seq:     (B, n_months, macro_dim)
    """
    def __init__(self,
                 merge_input_dim: int,
                 article_embedding_dim: int,
                 macro_dim: int,
                 output_dim: int,
                 fin_d_model: int = 128,
                 art_d_model: int = 128,
                 month_d_model: int = 256,
                 fin_layers: int = 2,
                 art_layers: int = 2,
                 month_layers: int = 2,
                 nhead: int = 4,
                 ff_mult: int = 4,
                 dropout_prob: float = 0.3):
        super().__init__()
        self.output_dim = output_dim

        # Daily -> monthly encoders
        self.financial_encoder = TransformerPooler(
            in_dim=merge_input_dim,
            d_model=fin_d_model,
            nhead=nhead,
            num_layers=fin_layers,
            dim_feedforward=ff_mult * fin_d_model,
            dropout=dropout_prob,
        )
        self.article_encoder = TransformerPooler(
            in_dim=article_embedding_dim,
            d_model=art_d_model,
            nhead=nhead,
            num_layers=art_layers,
            dim_feedforward=ff_mult * art_d_model,
            dropout=dropout_prob,
        )

        # Monthly encoder over time (sequence of months)
        monthly_in_dim = fin_d_model + art_d_model + macro_dim
        self.month_proj = nn.Linear(monthly_in_dim, month_d_model)
        self.monthly_encoder = TransformerPooler(
            in_dim=month_d_model,
            d_model=month_d_model,
            nhead=nhead,
            num_layers=month_layers,
            dim_feedforward=ff_mult * month_d_model,
            dropout=dropout_prob,
        )

        # Prediction head
        self.mlp_head = nn.Sequential(
            nn.Linear(month_d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, output_dim),
        )

    def _encode_monthly_from_daily(self, seq_4d: torch.Tensor, encoder: TransformerPooler):
        # seq_4d: (B, n_months, T, D)
        B, nM, T, D = seq_4d.shape
        x = seq_4d.reshape(B * nM, T, D)
        pooled = encoder(x)  # (B*nM, d_model)
        return pooled.view(B, nM, -1)

    def forward(self, financial_seq: torch.Tensor, article_seq: torch.Tensor, macro_seq: torch.Tensor):
        B, nM, days_per_month, _ = financial_seq.shape

        # Step 1: daily encoders -> per-month vectors
        fin_monthly = self._encode_monthly_from_daily(financial_seq, self.financial_encoder)  # (B, nM, F)

        # article_seq: (B, nM, L, D)
        B2, nM2, L, D2 = article_seq.shape
        assert (B2 == B and nM2 == nM), 'article_seq must align with financial_seq in (B, n_months)'
        art_monthly = self._encode_monthly_from_daily(article_seq, self.article_encoder)       # (B, nM, A)

        # Step 2: concat monthly features + macro
        monthly_feats = torch.cat([fin_monthly, art_monthly, macro_seq], dim=-1)               # (B, nM, F+A+M)
        monthly_feats = self.month_proj(monthly_feats)                                         # (B, nM, Md)

        # Step 3: transformer over months (returns CLS)
        months_repr = self.monthly_encoder(monthly_feats)                                      # (B, Md)

        # Step 4: head
        return self.mlp_head(months_repr)