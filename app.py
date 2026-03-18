"""
E-Commerce Customer Segmentation using K-Means Clustering
===========================================================
Unsupervised ML pipeline that segments real e-commerce customers
into distinct groups based on RFM (Recency, Frequency, Monetary) analysis.

Dataset: Online Retail Dataset (Kaggle - carrie1/ecommerce-data)
- Real UK e-commerce transactions 2010-2011
- 500,000+ transactions
- Features engineered: Recency, Frequency, Monetary Value, Avg Order Value

Tech Stack:
- Scikit-learn — K-Means Clustering, StandardScaler
- Pandas, NumPy — Data processing and RFM feature engineering
- Matplotlib, Seaborn — 9-panel visualization dashboard
- Elbow Method + Silhouette Score — Optimal cluster validation

Author: Gourav Yadav
"""

import os
import warnings
warnings.filterwarnings("ignore")

import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ══════════════════════════════════════════════
# LOAD & ENGINEER FEATURES
# ══════════════════════════════════════════════

def load_and_engineer(path):
    """
    Load raw transaction data and engineer RFM features:
    - Recency   : Days since last purchase
    - Frequency : Number of unique invoices
    - Monetary  : Total spend
    - AvgOrderValue : Average spend per invoice
    - UniqueProducts : Number of unique products bought
    """
    csv_path = os.path.join(path, "data.csv")
    print(f"Loading data from: {csv_path}")

    # Load with encoding fix
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    print(f"Raw transactions: {len(df)}")

    # Clean data
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    print(f"Clean transactions: {len(df)}")
    print(f"Unique customers: {df['CustomerID'].nunique()}")

    # Reference date for recency
    ref_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # RFM Feature Engineering
    print("Engineering RFM features...")
    rfm = df.groupby('CustomerID').agg(
        recency=('InvoiceDate', lambda x: (ref_date - x.max()).days),
        frequency=('InvoiceNo', 'nunique'),
        monetary=('TotalPrice', 'sum'),
        avg_order_value=('TotalPrice', 'mean'),
        unique_products=('StockCode', 'nunique')
    ).reset_index()

    # Remove outliers using IQR
    for col in ['monetary', 'frequency', 'avg_order_value']:
        Q1 = rfm[col].quantile(0.05)
        Q3 = rfm[col].quantile(0.95)
        rfm = rfm[(rfm[col] >= Q1) & (rfm[col] <= Q3)]

    print(f"Customers after cleaning: {len(rfm)}")
    print(f"\nRFM Summary:")
    print(rfm[['recency', 'frequency', 'monetary',
               'avg_order_value', 'unique_products']].describe().round(2).to_string())

    return rfm


# ══════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════

def preprocess(rfm):
    """Scale RFM features for clustering."""
    features = ['recency', 'frequency', 'monetary',
                'avg_order_value', 'unique_products']
    X = rfm[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\nFeatures scaled. Shape: {X_scaled.shape}")
    return X_scaled, scaler, features


# ══════════════════════════════════════════════
# OPTIMAL CLUSTERS
# ══════════════════════════════════════════════

def find_optimal_k(X_scaled, max_k=10):
    """Elbow Method + Silhouette Score."""
    print("\n" + "="*50)
    print("Finding optimal number of clusters...")
    print("="*50)

    inertias, silhouette_scores, k_range = [], [], range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        score = silhouette_score(X_scaled, km.labels_)
        silhouette_scores.append(score)
        print(f"  k={k}: Inertia={km.inertia_:.0f}, Silhouette={score:.4f}")

    best_k = list(k_range)[np.argmax(silhouette_scores)]
    print(f"\nBest k: {best_k} (Silhouette: {max(silhouette_scores):.4f})")
    return inertias, silhouette_scores, list(k_range), best_k


# ══════════════════════════════════════════════
# CLUSTERING
# ══════════════════════════════════════════════

def cluster(rfm, X_scaled, n_clusters):
    """Apply K-Means and label segments."""
    print(f"\nApplying K-Means with k={n_clusters}...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['cluster'] = km.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, rfm['cluster'])
    print(f"Silhouette Score: {score:.4f}")

    # Auto-label based on RFM characteristics
    summary = rfm.groupby('cluster').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean'
    })

    labels = {}
    for cid in rfm['cluster'].unique():
        r = summary.loc[cid, 'recency']
        f = summary.loc[cid, 'frequency']
        m = summary.loc[cid, 'monetary']

        med_r = summary['recency'].median()
        med_f = summary['frequency'].median()
        med_m = summary['monetary'].median()

        if r < med_r and f > med_f and m > med_m:
            labels[cid] = "Champions"
        elif r < med_r and m > med_m:
            labels[cid] = "Loyal Customers"
        elif r > med_r and f > med_f:
            labels[cid] = "At Risk"
        else:
            labels[cid] = "New Customers"

    # Handle duplicate labels
    used = {}
    final_labels = {}
    fallback = ["Segment A", "Segment B", "Segment C",
                "Segment D", "Segment E"]
    fb_idx = 0
    for cid, label in labels.items():
        if label in used.values():
            final_labels[cid] = fallback[fb_idx]
            fb_idx += 1
        else:
            final_labels[cid] = label
        used[cid] = final_labels[cid]

    rfm['segment'] = rfm['cluster'].map(final_labels)

    print("\nSegment Summary:")
    seg_summary = rfm.groupby('segment').agg(
        customers=('CustomerID', 'count'),
        avg_recency=('recency', 'mean'),
        avg_frequency=('frequency', 'mean'),
        avg_monetary=('monetary', 'mean')
    ).round(2)
    print(seg_summary.to_string())

    return rfm, km, score, final_labels


# ══════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════

def visualize(rfm, features, inertias, silhouette_scores,
              k_range, n_clusters, X_scaled):
    """9-panel visualization dashboard."""
    print("\nGenerating visualizations...")

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12',
              '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    seg_color_map = {seg: colors[i] for i, seg in
                     enumerate(rfm['segment'].unique())}

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('E-Commerce Customer Segmentation — RFM Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. Elbow Method
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=n_clusters, color='red', linestyle='--',
                label=f'k={n_clusters}')
    ax1.set_title('Elbow Method', fontweight='bold')
    ax1.set_xlabel('k'); ax1.set_ylabel('Inertia')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # 2. Silhouette Scores
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=n_clusters, color='red', linestyle='--',
                label=f'k={n_clusters}')
    ax2.set_title('Silhouette Scores', fontweight='bold')
    ax2.set_xlabel('k'); ax2.set_ylabel('Score')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # 3. Segment Distribution
    ax3 = fig.add_subplot(3, 3, 3)
    counts = rfm['segment'].value_counts()
    pie_colors = [seg_color_map.get(s, '#95a5a6') for s in counts.index]
    ax3.pie(counts.values, labels=counts.index, colors=pie_colors,
            autopct='%1.1f%%', startangle=90)
    ax3.set_title('Customer Distribution', fontweight='bold')

    # 4. Recency vs Monetary
    ax4 = fig.add_subplot(3, 3, 4)
    for seg in rfm['segment'].unique():
        mask = rfm['segment'] == seg
        ax4.scatter(rfm[mask]['recency'], rfm[mask]['monetary'],
                   c=seg_color_map[seg], label=seg, alpha=0.6, s=40)
    ax4.set_xlabel('Recency (days)')
    ax4.set_ylabel('Monetary (£)')
    ax4.set_title('Recency vs Monetary', fontweight='bold')
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    # 5. Frequency vs Monetary
    ax5 = fig.add_subplot(3, 3, 5)
    for seg in rfm['segment'].unique():
        mask = rfm['segment'] == seg
        ax5.scatter(rfm[mask]['frequency'], rfm[mask]['monetary'],
                   c=seg_color_map[seg], label=seg, alpha=0.6, s=40)
    ax5.set_xlabel('Frequency (orders)')
    ax5.set_ylabel('Monetary (£)')
    ax5.set_title('Frequency vs Monetary', fontweight='bold')
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

    # 6. Avg Order Value by Segment
    ax6 = fig.add_subplot(3, 3, 6)
    aov = rfm.groupby('segment')['avg_order_value'].mean().sort_values(ascending=False)
    bar_colors = [seg_color_map.get(s, '#95a5a6') for s in aov.index]
    bars = ax6.bar(aov.index, aov.values, color=bar_colors, edgecolor='white')
    ax6.set_title('Avg Order Value by Segment', fontweight='bold')
    ax6.set_xlabel('Segment'); ax6.set_ylabel('£')
    ax6.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, aov.values):
        ax6.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'£{val:.0f}', ha='center', fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Recency Distribution
    ax7 = fig.add_subplot(3, 3, 7)
    for seg in rfm['segment'].unique():
        mask = rfm['segment'] == seg
        ax7.hist(rfm[mask]['recency'], bins=20, alpha=0.6,
                label=seg, color=seg_color_map[seg])
    ax7.set_title('Recency Distribution', fontweight='bold')
    ax7.set_xlabel('Days Since Last Purchase')
    ax7.set_ylabel('Count')
    ax7.legend(fontsize=8); ax7.grid(True, alpha=0.3)

    # 8. RFM Heatmap
    ax8 = fig.add_subplot(3, 3, 8)
    hmap = rfm.groupby('segment')[['recency', 'frequency',
                                    'monetary', 'avg_order_value',
                                    'unique_products']].mean()
    hmap_norm = (hmap - hmap.min()) / (hmap.max() - hmap.min())
    sns.heatmap(hmap_norm.T, annot=True, fmt='.2f',
                cmap='YlOrRd', ax=ax8, linewidths=0.5)
    ax8.set_title('RFM Feature Heatmap\n(Normalized)', fontweight='bold')

    # 9. PCA 2D
    ax9 = fig.add_subplot(3, 3, 9)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    for seg in rfm['segment'].unique():
        mask = rfm['segment'] == seg
        ax9.scatter(X_pca[mask.values, 0], X_pca[mask.values, 1],
                   c=seg_color_map[seg], label=seg, alpha=0.6, s=40)
    ax9.set_xlabel(f'PC1 ({explained[0]:.1%})')
    ax9.set_ylabel(f'PC2 ({explained[1]:.1%})')
    ax9.set_title('PCA 2D Visualization', fontweight='bold')
    ax9.legend(fontsize=8); ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('customer_segmentation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: customer_segmentation.png")


# ══════════════════════════════════════════════
# BUSINESS RECOMMENDATIONS
# ══════════════════════════════════════════════

def recommendations(rfm):
    """Print business recommendations per segment."""
    print("\n" + "="*50)
    print("BUSINESS RECOMMENDATIONS")
    print("="*50)

    recs = {
        "Champions": [
            "Offer VIP membership and exclusive rewards",
            "Early access to new products and sales",
            "Referral program — they are your best advocates"
        ],
        "Loyal Customers": [
            "Upsell premium products and bundles",
            "Membership upgrade incentives",
            "Personalized email campaigns"
        ],
        "At Risk": [
            "Send win-back email with special discount",
            "Survey to understand why they stopped buying",
            "Limited time offer to trigger purchase"
        ],
        "New Customers": [
            "Welcome series with product education",
            "First purchase discount on next order",
            "Show bestsellers and top-rated products"
        ]
    }

    for seg in rfm['segment'].unique():
        count = len(rfm[rfm['segment'] == seg])
        print(f"\n{seg} ({count} customers)")
        actions = recs.get(seg, ["Analyze segment behavior and create targeted strategy"])
        for action in actions:
            print(f"  • {action}")


# ══════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════

def export(rfm):
    """Export results to CSV."""
    cols = ['CustomerID', 'recency', 'frequency', 'monetary',
            'avg_order_value', 'unique_products', 'cluster', 'segment']
    rfm.to_csv('customer_segments.csv', index=False,
               columns=[c for c in cols if c in rfm.columns])
    print(f"\nExported: customer_segments.csv ({len(rfm)} customers)")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
print("yo")
if __name__ == "__main__":
    print("started")

    import sys 
    sys.stdout.flush()
    print("hiiii")
    print("="*50)
    print("E-COMMERCE CUSTOMER SEGMENTATION")
    print("Dataset: Online Retail (Kaggle)")
    print("="*50)

    # Download dataset
    print("\nDownloading dataset from Kaggle...")
    path = kagglehub.dataset_download("carrie1/ecommerce-data")
    print(f"Dataset path: {path}")

    # Load and engineer features
    rfm = load_and_engineer(path)

    # Preprocess
    X_scaled, scaler, features = preprocess(rfm)

    # Find optimal k
    inertias, sil_scores, k_range, best_k = find_optimal_k(X_scaled)

    # Cluster
    rfm, km, score, labels = cluster(rfm, X_scaled, best_k)

    # Visualize
    visualize(rfm, features, inertias, sil_scores,
              k_range, best_k, X_scaled)

    # Recommendations
    recommendations(rfm)

    # Export
    export(rfm)

    print("\n" + "="*50)
    print("DONE!")
    print(f"Silhouette Score: {score:.4f}")
    print(f"Segments: {best_k}")
    print("Output: customer_segmentation.png")
    print("Output: customer_segments.csv")
    print("="*50)