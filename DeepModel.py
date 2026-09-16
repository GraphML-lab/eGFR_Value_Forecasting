from torchinfo import summary

Hidden_Dim = 256
embed_Ftr_dim = 256

# =========================
# Masking Function
# =========================
def apply_mask(x, mask_ratio=0.2):
    """
    x: (batch, seq_len, num_features)
    """
    x_masked = x.clone()

    # ---- feature-level masking ----
    mask = torch.rand_like(x) < mask_ratio
    x_masked[mask] = 0.0

    if np.random.rand() < 0.3:
        t = np.random.randint(0, x.shape[1])
        x_masked[:, t, :] = 0.0

        mask[:, t, :] = True

    #print(x.shape)    
    #print(x_masked.shape)    
    return x_masked, mask


# =========================
# Attention Pooling
# =========================
class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (B, T, D)
        weights = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        pooled = (x * weights).sum(dim=1)             # (B, D)
        return pooled


# =========================================================
# Temporal Conv Block
# =========================================================

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: (B, T, F) → (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        # back to (B, T, F)
        return x.permute(0, 2, 1)


# =========================================================
# Gated Attention Pooling
# =========================================================

class GatedAttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):

        # x: (B, T, D)

        gate = self.gate(x)

        x = x * gate

        scores = self.attn(x)

        weights = torch.softmax(scores, dim=1)

        pooled = torch.sum(weights * x, dim=1)

        return pooled


# =========================================================
# Improved Temporal Encoder
# =========================================================

class DeepModel(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=Hidden_Dim,
        embed_dim=embed_Ftr_dim,
        num_heads=4,
        dropout=0.2
    ):

        super().__init__()

        # =================================================
        # TCN Branch
        # =================================================

        self.tcn1 = TemporalConvBlock(input_dim, hidden_dim, kernel_size=1)
        self.tcn3 = TemporalConvBlock(input_dim, hidden_dim, kernel_size=3)
        self.tcn5 = TemporalConvBlock(input_dim, hidden_dim, kernel_size=5)

        self.tcn2 = TemporalConvBlock(hidden_dim,  hidden_dim)
        
        total_dim = hidden_dim*7
        self.tcn_fusion = TemporalConvBlock(total_dim,  hidden_dim)

        
        # =================================================
        # BiGRU Branch
        # =================================================

        self.bigru = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True
        )
        self.bigru1 = nn.GRU(
            input_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True
        )
        self.bigru2 = nn.GRU(
            input_dim,
            hidden_dim // 2 // 2,
            batch_first=True,
            bidirectional=True
        )
        self.bigru41 = nn.GRU(
            input_dim,
            hidden_dim // 2 // 4,
            batch_first=True,
            bidirectional=True
        )
        self.bigru42 = nn.GRU(
            input_dim,
            hidden_dim // 1 // 4,
            batch_first=True,
            bidirectional=False
        )

        # =================================================
        # Fusion
        # =================================================

        self.fusion = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        # =================================================
        # Lightweight Self Attention
        # =================================================

        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        # =================================================
        # LayerNorm
        # =================================================

        self.norm1 = nn.LayerNorm(hidden_dim)

        # =================================================
        # Pooling
        # =================================================

        #self.pool = GatedAttentionPooling(hidden_dim)
        self.pool = AttentionPooling(hidden_dim)

        # =================================================
        # Embedding
        # =================================================

        self.embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # =================================================
        # Better Decoder
        # =================================================

        self.decoder_gru = nn.GRU(
            embed_dim,
            hidden_dim,
            batch_first=True
        )

        self.decoder_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.dropout = nn.Dropout(dropout)

        
        
        
        self.backbone = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),       nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),       nn.GELU(), nn.Dropout(dropout),
        )
 
        # ── Step 6: Output heads ──
        self.reg_head = nn.Linear(embed_dim, 2)
    # =====================================================
    # Forward
    # =====================================================

    def forward(self, x):

        # x: (B, T, F)

        B, T, Fdim = x.shape

        # -------------------------------------------------
        # TCN branch
        # -------------------------------------------------

        tcn = self.tcn1(x)
        tcn1 = self.dropout(tcn)

        tcn = self.tcn3(x)
        tcn3 = self.dropout(tcn)

        tcn = self.tcn5(x)
        tcn5 = self.dropout(tcn)

        tcn = self.tcn3(x)
        tcn = self.dropout(tcn)
        tcn,_ = self.bigru(tcn)
        tcn3gru = self.dropout(tcn)

        tcn = self.tcn5(x)
        tcn = self.dropout(tcn)
        tcn,_ = self.bigru(tcn)
        tcn5gru = self.dropout(tcn)


        # -------------------------------------------------
        # GRU branch
        # -------------------------------------------------

        gru, _ = self.bigru1(x)
        gru1 = self.dropout(gru)

        gru, _ = self.bigru2(x)
        gru2 = self.dropout(gru)

        gru, _ = self.bigru41(x)
        gru41 = self.dropout(gru)

        gru, _ = self.bigru42(x)
        gru42 = self.dropout(gru)


        # -------------------------------------------------
        # Fusion
        # -------------------------------------------------
        fused = torch.cat([tcn1, tcn3, tcn5, tcn3gru, tcn5gru, gru1, gru2, gru41, gru42], dim=-1)
        
        tcn = self.tcn_fusion(fused)
        fused = self.dropout(tcn)


        # -------------------------------------------------
        # Self Attention
        # -------------------------------------------------

        fused, _ = self.mha(fused, fused, fused)


        # -------------------------------------------------
        # Pooling
        # -------------------------------------------------

        pooled = self.pool(fused)

        # -------------------------------------------------
        # Embedding
        # -------------------------------------------------

        emb = self.embedding(pooled)

        feat = self.backbone(emb)                              # (B, 128)
 
        # ── Output heads ──
        reg     = self.reg_head(feat)
        mean    = reg[:, 0]
        log_var = reg[:, 1]

        return mean, log_var, feat

    
    
    
    
    
model = DeepModel(
        input_dim=num_features,
        hidden_dim=Hidden_Dim,
        embed_dim=embed_Ftr_dim,
        num_heads=4,
        dropout=0.45
)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")



summary(model, input_size=(1024, T, X_train.shape[-1]))
