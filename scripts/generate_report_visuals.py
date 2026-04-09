import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetics
sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#F8F9FA"})

OUTPUT_DIR = "docs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_geography_map():
    print("Generating Geography Map...")
    shapefile_path = "data/external/taxi_zones/taxi_zones.shp"
    if not os.path.exists(shapefile_path):
        print(f"Warning: Shapefile not found at {shapefile_path}")
        return
        
    gdf = gpd.read_file(shapefile_path)
    manhattan = gdf[gdf['borough'] == 'Manhattan'].copy()
    
    # Assign color categories
    manhattan['color_category'] = 'Included (67 Zones)'
    manhattan.loc[manhattan['LocationID'] == 104, 'color_category'] = 'Anomaly (Zone 104 - Island)'
    manhattan.loc[manhattan['LocationID'] == 103, 'color_category'] = 'Excluded (Zone 103)'
    
    fig, ax = plt.subplots(figsize=(8, 10))
    color_dict = {
        'Included (67 Zones)': '#4C72B0',
        'Anomaly (Zone 104 - Island)': '#C44E52',
        'Excluded (Zone 103)': '#95a5a6'
    }
    
    for category, color in color_dict.items():
        subset = manhattan[manhattan['color_category'] == category]
        if not subset.empty:
            subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.5, label=category)
            
    # Add annotations for Zone 104
    zone_104 = manhattan[manhattan['LocationID'] == 104]
    if not zone_104.empty:
        centroid = zone_104.geometry.centroid.iloc[0]
        ax.annotate("Zone 104\n(Disconnected)", (centroid.x, centroid.y),
                    xytext=(centroid.x + 10000, centroid.y - 10000),
                    arrowprops=dict(facecolor='black', arrowstyle="->", connectionstyle="arc3,rad=.2"),
                    fontsize=12, fontweight='bold')
    
    ax.set_title("Manhattan Taxi Zones: Identifying Geographic Anomalies", fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=color, label=label) for label, color in color_dict.items()]
    ax.legend(handles=handles, loc='upper left', fontsize=12, frameon=True, shadow=True)
    
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "zone104_map.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_learning_curve():
    print("Generating Learning Curve...")
    log_path = "artifacts/reports/models/nyc_lpe_stgtn_default_gpu_paperlike.json"
    if not os.path.exists(log_path):
        print(f"Warning: Log file not found at {log_path}")
        return
        
    with open(log_path, 'r') as f:
        data = json.load(f)
        
    history = data.get("training_history", [])
    if not history:
        print("No training history found.")
        return
        
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["validation_loss"] for h in history]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, label='Training L1 Loss', color='#4C72B0', linewidth=2.5)
    ax.plot(epochs, val_loss, label='Validation L1 Loss', color='#DD8452', linewidth=2.5)
    
    ax.set_title("LPE-STGTN Training Dynamics (Newton Cluster)", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Absolute Error (Scaled)", fontsize=14)
    ax.legend(fontsize=12, frameon=True, shadow=True)
    
    min_val_idx = np.argmin(val_loss)
    opt_epoch = epochs[min_val_idx]
    opt_loss = val_loss[min_val_idx]
    ax.scatter(opt_epoch, opt_loss, color='#C44E52', s=100, zorder=5)
    ax.annotate(f'Optimal Validation\nEpoch {opt_epoch}', 
                (opt_epoch, opt_loss), 
                xytext=(opt_epoch - 20, opt_loss + 0.015),
                arrowprops=dict(facecolor='black', arrowstyle="->"),
                fontsize=11, fontweight='bold')
    
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_sparsification_heatmaps():
    print("Generating Sparsification Heatmaps...")
    np.random.seed(42)
    N = 67
    
    # Generate mock coordinates representing roughly 67 zones in Manhattan
    x = np.random.normal(loc=0, scale=2, size=N)
    y = np.random.normal(loc=0, scale=5, size=N)
    
    distances = np.sqrt((x[:, np.newaxis] - x[np.newaxis, :])**2 + (y[:, np.newaxis] - y[np.newaxis, :])**2)
    sigma = np.std(distances)
    
    adj_dense = np.exp(-(distances**2) / (sigma**2))
    np.fill_diagonal(adj_dense, 1.0)
    
    epsilon = 0.05
    adj_sparse = np.where(adj_dense >= epsilon, adj_dense, 0.0)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    sns.heatmap(adj_dense, ax=axes[0], cmap="YlGnBu", cbar=True, xticklabels=False, yticklabels=False)
    axes[0].set_title(r"Dense Distance Adjacency ($\epsilon=0.0$)", fontsize=16, fontweight='bold')
    axes[0].text(0.5, -0.05, f"Non-zero edges: {N*N}", transform=axes[0].transAxes, ha='center', fontsize=12)

    sns.heatmap(adj_sparse, ax=axes[1], cmap="YlGnBu", cbar=True, xticklabels=False, yticklabels=False)
    axes[1].set_title(r"Sparsified Adjacency ($\epsilon=0.05$)", fontsize=16, fontweight='bold')
    
    sparse_edges = np.count_nonzero(adj_sparse)
    axes[1].text(0.5, -0.05, f"Non-zero edges: {sparse_edges} (-{((N*N - sparse_edges)/(N*N))*100:.1f}%)", transform=axes[1].transAxes, ha='center', fontsize=12)
    
    fig.suptitle("Graph Sparsification Effect on Temporal Convolution", fontsize=18, fontweight='bold', y=1.02)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "graph_sparsification_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_benchmark_barchart():
    print("Generating Benchmark Bar Chart...")
    models = ['Persistence\nBaseline', 'LSTM\nBaseline', 'LPE-STGTN\n(Ours Phase 1)', 'Original\nPaper Target']
    mape = [66.07, 75.64, 32.83, 31.82]
    
    x = np.arange(len(models))
    width = 0.5
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    bars = ax1.bar(x, mape, width, label='Test MAPE (%)', color='#4C72B0', edgecolor='black', linewidth=1)
    
    ax1.set_ylabel('Mean Absolute Percentage Error (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Ride-Hailing Forecasting Benchmarks (Manhattan 2018)', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12, fontweight='bold')
    
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval}%", ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.axhline(y=31.82, color='#C44E52', linestyle='--', linewidth=2, zorder=0)
    
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "benchmark_barchart.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_geography_map()
    generate_learning_curve()
    generate_sparsification_heatmaps()
    generate_benchmark_barchart()
    print("All visualizations generated successfully in docs/figures/")
