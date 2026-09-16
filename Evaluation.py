
def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = "cpu"):
    model.eval()
    all_means, all_stds, all_targets, all_gates = [], [], [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            mean, log_var, _ = model(xb)
            std = torch.exp(0.5 * log_var)
            all_means.append(mean.cpu().numpy())
            all_stds.append(std.cpu().numpy())
            all_targets.append(yb.numpy())
            #all_gates.append(gates.cpu().numpy())

    preds   = np.concatenate(all_means)
    stds    = np.concatenate(all_stds)
    targets = np.concatenate(all_targets)
    #gates   = np.concatenate(all_gates)
    gates   = 0
    
    print(f"targets: {targets[:10]}")

    mae  = mean_absolute_error(targets, preds)
    rmse = np.sqrt(np.mean((targets - preds) ** 2))
    r2   = r2_score(targets, preds)
    # 95% prediction interval coverage
    within_95 = np.mean(np.abs(targets - preds) <= 1.96 * stds)

    print("\n══════════════ Deep Model Results ══════════════")
    print(f"  MAE              : {mae:.4f}")
    print(f"  RMSE             : {rmse:.4f}")
    print(f"  R²               : {r2:.4f}")

    
    
    
    
    maeBL  = mean_absolute_error(targets, baseline_egfr_test)
    rmseBL = np.sqrt(np.mean((targets - baseline_egfr_test) ** 2))
    r2BL   = r2_score(targets, baseline_egfr_test)
    # 95% prediction interval coverage
    within_95BL = np.mean(np.abs(targets - baseline_egfr_test) <= 1.96 * stds)

    print("\n════════════ Baseline Results ════════════")
    print(f"  MAE              : {maeBL:.4f}")
    print(f"  RMSE             : {rmseBL:.4f}")
    print(f"  R²               : {r2BL:.4f}")

    
    


# Structure the data matrix
    results_data = {
    "Metric": ["MAE", "RMSE", "R²", "95% PI Coverage"],
    "EEstimated": [
        f"{mae:.4f}", 
        f"{rmse:.4f}", 
        f"{r2:.4f}", 
        f"{within_95 * 100:.1f}%"
        
    ],
    "Baseline": [
        f"{maeBL:.4f}", 
        f"{rmseBL:.4f}", 
        f"{r2BL:.4f}", 
        f"{within_95BL * 100:.1f}%"
    ]
    }

# Create DataFrame and export to CSV
    df_results = pd.DataFrame(results_data)
    df_results.to_csv(f"../eGFR_value_estimation/outputs/results_v02_horiz-{HORIZON}_fold-{iteration}.csv", index=False)    
    
    
    
    return preds, stds, targets, gates





preds, stds, targets, gates = evaluate_model(model, test_loader, device=DEVICE)
