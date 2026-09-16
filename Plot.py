n_bins = 120

def plot_results(history, preds, stds, targets, gates):
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle("Multi-Scale — eGFR Prediction", fontsize=14, fontweight="bold")

    # ── A: Loss curves ──
    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train Loss")
    ax.plot(history["val_loss"],   label="Val Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim([min(np.min(history["train_loss"]), np.min(history["val_loss"]))*0.9//1  ,  history["train_loss"][2]*1.2])

    # ── B1: Predicted vs Actual ──
    ax = axes[0, 1]
    ax.scatter(targets, preds, alpha=0.1, s=5, color="steelblue")
    lo, hi = targets.min(), targets.max()
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Ideal")
    ax.set_xlabel("Actual eGFR"); ax.set_ylabel("Predicted eGFR")
    ax.set_title("Predicted vs Actual"); ax.legend(); ax.grid(True, alpha=0.3)

    # ── B2: baseline vs Actual ──
    ax = axes[1, 1]
    ax.scatter(targets, baseline_egfr_test, alpha=0.1, s=5, color="steelblue")
    lo, hi = targets.min(), targets.max()
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Ideal")
    ax.set_xlabel("Actual eGFR"); ax.set_ylabel("Predicted eGFR")
    ax.set_title("Predicted vs Actual"); ax.legend(); ax.grid(True, alpha=0.3)


    # ── C: Uncertainty (first 100 samples) ──
    ax = axes[1, 0]
    idx = np.arange(min(100, len(preds)))
    sort_i = np.argsort(targets[idx])
    ax.plot(targets[idx][sort_i], label="Actual",    color="black",    linewidth=1.2)
    ax.plot(preds[idx][sort_i],   label="Predicted", color="steelblue", linewidth=1.2)
    ax.fill_between(
        np.arange(len(sort_i)),
        preds[idx][sort_i] - 1.96 * stds[idx][sort_i],
        preds[idx][sort_i] + 1.96 * stds[idx][sort_i],
        alpha=0.25, color="steelblue", label="95% PI",
    )
    ax.set_xlabel("Sample (sorted by actual)"); ax.set_ylabel("eGFR")
    ax.set_title("Predictions with Uncertainty"); ax.legend(); ax.grid(True, alpha=0.3)



    
    difference = targets - preds
    # Plotting the distribution
    import pandas as pd
    try:
        pd._config.config.register_option('mode.use_inf_as_null', False)
        print("Successfully patched Pandas option for Seaborn compatibility.")
    except Exception:
        pass    #if pd.get_option('mode.use_inf_as_null'):
    
    ax = axes[2, 0]
    #sns.histplot(difference, kde=True, ax=ax, color='skyblue', edgecolor='black', bins=n_bins)
    sns.histplot(difference, kde=True, ax=ax, color='skyblue', bins=n_bins, fill=False, linewidth=0)
    
# Add titles and labels
    ax.set_title('Distribution of Difference between Targets and Predictions', fontsize=14)
    ax.set_xlabel('Difference (Targets - Predictions)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_xlim([-100, 100])
# Add a vertical reference line at 0 error
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
    ax.legend()
    
    
    


# 1. Calculate both differences
    # difference_preds = targets - preds
    # difference_baseline = targets - baseline_egfr_test
    
    valid_mask = ~np.isnan(preds) & ~np.isnan(baseline_egfr_test) & ~np.isnan(targets)

    difference_preds = (targets - preds)[valid_mask]
    difference_baseline = (targets - baseline_egfr_test)[valid_mask]

# 2. Select the single axis where you want the combined plot
    ax = axes[2, 1] 

# 3. Plot the first distribution (Model Predictions)
    sns.histplot(
    difference_preds, 
    kde=False, 
    ax=ax, 
    color='blue', 
    element="step",
    #edgecolor='black', 
    bins=n_bins, 
    fill=False, linewidth=2, 
    label='Model Predictions',
    alpha=0.6  # Adds transparency so overlapping bars are visible
    )

# 4. Plot the second distribution (Baseline) on the SAME axis
    sns.histplot(
    difference_baseline, 
    kde=False, 
    ax=ax, 
    color='orange', # Different color to distinguish it
    #edgecolor='black', 
    bins=n_bins, 
    fill=False, linewidth=2, 
    label='Baseline',
    alpha=0.6  # Adds transparency
    )

# 5. Add titles, labels, and reference line once for the combined plot
    ax.set_title('Distribution of Differences from Targets', fontsize=14)
    ax.set_xlabel('Difference (Targets - Predictions / Baseline)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_xlim([-100, 100])

# Add a vertical reference line at 0 error
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')

# ax.legend() will automatically catch the labels from both histplots and the axvline
    ax.legend()






# Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f"../eGFR_value_estimation/outputs/plots_v02_horiz-{HORIZON}_fold-{iteration}_deep_50.png", dpi=400, bbox_inches="tight")
    plt.show()
    print("Figure saved → scr_results.png")
    

    
    
    
    

    
plot_results(history, preds, stds, targets, gates)
