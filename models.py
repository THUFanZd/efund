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
    

class PositionalEncoder(nn.Module):
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
        self.pe = PositionalEncoder(d_model, dropout)
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
class EconomicIndicatorPredictorTrmDecoder(nn.Module):
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
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000, batch_first: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)  # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:, :pe[:, 1::2].shape[1]]
        pe = pe.unsqueeze(0)  # (1, L, D) for batch_first
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        # x: (B, L, D) if batch_first
        if self.batch_first:
            L = x.size(1)
            x = x + self.pe[:, :L, :]
        else:
            L = x.size(0)
            x = x + self.pe[:L, :].unsqueeze(1)
        return self.dropout(x)

class EconomicIndicatorPredictorTrm(nn.Module):
    def __init__(self, 
                 merge_input_dim,
                 article_embedding_dim,
                 macro_dim,
                 output_dim,
                 d_model=128,
                 nhead=8,
                 enc_num_layers=2,
                 dec_num_layers=2,
                 dim_feedforward=256,
                 dropout_prob=0.3):
        super().__init__()

        self.d_model = d_model

        # --- Daily financial encoder ---
        self.financial_in = nn.Linear(merge_input_dim, d_model)
        self.financial_pos = PositionalEncoding(d_model, dropout_prob, batch_first=True)
        enc_layer_f = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout_prob, batch_first=True)
        self.financial_encoder = nn.TransformerEncoder(enc_layer_f, num_layers=enc_num_layers)
        self.dropout_financial = nn.Dropout(dropout_prob)

        # --- Article encoder (per month) ---
        self.article_in = nn.Linear(article_embedding_dim, d_model)
        self.article_pos = PositionalEncoding(d_model, dropout_prob, batch_first=True)
        enc_layer_a = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout_prob, batch_first=True)
        self.article_encoder = nn.TransformerEncoder(enc_layer_a, num_layers=enc_num_layers)
        self.dropout_article = nn.Dropout(dropout_prob)

        # --- Monthly fusion projection ---
        # monthly_input_dim = financial_vec(d_model) + article_vec(d_model) + macro(macro_dim)
        monthly_input_dim = d_model + d_model + macro_dim
        self.monthly_proj = nn.Linear(monthly_input_dim, d_model)

        # --- Monthly Transformer Decoder ---
        self.monthly_pos = PositionalEncoding(d_model, dropout_prob, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_feedforward,
                                               dropout=dropout_prob, batch_first=True)
        self.monthly_decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_num_layers)

        # learnable query token for sequence-level decoding (tgt length = 1)
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.dropout_monthly = nn.Dropout(dropout_prob)

        # --- Output head ---
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, output_dim)
        )

    def _encode_per_month(self, seq_4d, in_linear, pos_enc, encoder):
        """
        seq_4d: (B, n_months, L, D_in)
        returns: (B, n_months, d_model)  # mean pooled encoder outputs
        """
        B, N, L, _ = seq_4d.shape
        x = seq_4d.view(B * N, L, -1)              # (B*N, L, D_in)
        x = in_linear(x)                           # (B*N, L, d_model)
        x = pos_enc(x)                             # add positional enc
        x = encoder(x)                             # (B*N, L, d_model)
        x = x.mean(dim=1)                          # mean pool over L -> (B*N, d_model)
        x = x.view(B, N, self.d_model)             # (B, N, d_model)
        return x

    def forward(self, financial_seq, article_seq, macro_seq):
        """
        financial_seq: (B, n_months, days_per_month, merge_input_dim)
        article_seq:   (B, n_months, L_art, article_embedding_dim)
        macro_seq:     (B, n_months, macro_dim)
        """
        B, N, _, _ = financial_seq.shape

        # Step 1: encode financial (per month)
        financial_monthly = self._encode_per_month(financial_seq,
                                                   self.financial_in,
                                                   self.financial_pos,
                                                   self.financial_encoder)
        financial_monthly = self.dropout_financial(financial_monthly)

        # Step 2: encode article (per month)
        article_monthly = self._encode_per_month(article_seq,
                                                 self.article_in,
                                                 self.article_pos,
                                                 self.article_encoder)
        article_monthly = self.dropout_article(article_monthly)

        # Step 3: concat + project to d_model as monthly memory
        monthly_features = torch.cat([financial_monthly, article_monthly, macro_seq], dim=-1)  # (B, N, d_f + d_a + m)
        memory = self.monthly_proj(monthly_features)  # (B, N, d_model)
        memory = self.monthly_pos(memory)             # add month-wise positions

        # Step 4: Transformer Decoder over months with a single query token
        # Create/expand query token for this batch
        query = self.query_token.expand(B, 1, self.d_model)  # (B, 1, d_model)
        # Optionally, you can add a (length-1) positional encoding to tgt as well:
        query = self.monthly_pos(query)  # still fine for length=1

        # No causal mask needed because tgt length is 1; memory is full context
        dec_out = self.monthly_decoder(tgt=query, memory=memory)  # (B, 1, d_model)
        seq_repr = self.dropout_monthly(dec_out.squeeze(1))       # (B, d_model)

        # Step 5: output
        output = self.mlp_head(seq_repr)  # (B, output_dim)
        return output
