 
class EGFRWeightedLoss(nn.Module):
    def __init__(self, alpha=0.2, focus_below=90.0):
        super().__init__()
        self.alpha = alpha         
        self.focus_below = focus_below

    def egfr_weight(self, targets):
        w = 1.0 
        return w  # shape: (batch,)

    def forward(self, mean, log_var, targets):
        w = self.egfr_weight(targets)             # (batch,)

        nll = (mean - targets) ** 2
        reg_loss = (w * nll).mean()

        total = reg_loss 
        return total, reg_loss.item(), 0



# ─────────────────────────────────────────────
# 8.  Training Loop
# ─────────────────────────────────────────────
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 400,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    patience: int = 20,
):
    model.to(device)
    criterion = EGFRWeightedLoss(alpha=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    history = {"train_loss": [], "val_loss": [], "val_mae": []}
    best_val  = float("inf")
    best_state = None
    no_improve = 0

    print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}  {'Val MAE':>9}  {'Reg':>8}  {'Aux':>8}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        # ── Training ──
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            mean, log_var, _ = model(xb)
            #print(f"yb1: {yb}")
            yb = yb.float() 
                
            if np.isnan(mean.cpu().detach().numpy()).any() or np.isnan(yb.cpu().detach().numpy()).any():
                print(f"NaN detected!")
                print(f"Number of NaNs in preds: {np.isnan(mean.cpu().detach().numpy()).sum()}")
                print(f"Number of NaNs in targets: {np.isnan(yb.cpu().detach().numpy()).sum()}")
                print(f"Max prediction value before crash: {np.nanmax(mean.cpu().detach().numpy())}")
                print(f"Min prediction value before crash: {np.nanmin(mean.cpu().detach().numpy())}")
                    
            #print(f"yb2: {yb}")
            loss, _, _ = criterion(mean, log_var, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                mean, log_var, _ = model(xb)
                
                if np.isnan(mean.cpu().detach().numpy()).any() or np.isnan(yb.cpu().detach().numpy()).any():
                    print(f"NaN detected!")
                    print(f"Number of NaNs in preds: {np.isnan(mean.cpu().detach().numpy()).sum()}")
                    print(f"Number of NaNs in targets: {np.isnan(yb.cpu().detach().numpy()).sum()}")
                    print(f"Max prediction value before crash: {np.nanmax(mean.cpu().detach().numpy())}")
                    print(f"Min prediction value before crash: {np.nanmin(mean.cpu().detach().numpy())}")
                    
                loss, reg_l, aux_l = criterion(mean, log_var, yb)
                val_loss += loss.item() * xb.size(0)
                all_preds.append(mean.cpu().numpy())
                all_targets.append(yb.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        
    
    
        preds   = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        val_mae = mean_absolute_error(targets, preds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        print(f"{epoch:>6}  {train_loss:>11.4f}  {val_loss:>10.4f}  {val_mae:>9.4f}"
              f"  {reg_l:>8.4f}  {aux_l:>8.4f}")

        # ── Early Stopping ──
        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    return model, history





model, history = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LR, device=DEVICE, patience=30 
)

