import matplotlib.pyplot as plt
import numpy as np

def generate_comparative_chart():
    # Data
    metrics = ['Semantic Similarity', 'Completeness']
    rag_scores = [0.880, 0.925]
    vanilla_scores = [0.733, 0.438]

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(metrics))
    width = 0.35  # width of the bars

    # Create bars
    rects1 = ax.bar(x - width/2, rag_scores, width, label='RAG (GPT-4o-mini)', color='#2ca02c')
    rects2 = ax.bar(x + width/2, vanilla_scores, width, label='Baseline (GPT-4o-mini)', color='#7f7f7f')

    # Add text, labels, and titles
    ax.set_ylabel('Score (0.0 to 1.0)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: Proposed RAG vs. Baseline GPT-4o-mini', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11, loc='upper right')

    # Function to attach a text label above each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    # Add subtle gridlines for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Save the figure
    plt.tight_layout()
    plt.savefig('comparative_metrics.png', dpi=300)
    print("[+] Saved comparative_metrics.png")
    plt.close()

def generate_rag_exclusive_chart():
    # Data
    metrics = ['Context Precision', 'Context Recall', 'Faithfulness']
    rag_scores = [1.000, 1.000, 0.608]

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(metrics))

    # Create bars
    rects = ax.bar(x, rag_scores, 0.5, color='#1f77b4')

    # Add text, labels, and titles
    ax.set_ylabel('Score (0.0 to 1.0)', fontsize=12, fontweight='bold')
    ax.set_title('RAG Pipeline Evaluation Metrics', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.15) # Slightly higher upper limit for the 1.000 labels

    # Attach labels
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add subtle gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Save the figure
    plt.tight_layout()
    plt.savefig('rag_exclusive_metrics.png', dpi=300)
    print("[+] Saved rag_exclusive_metrics.png")
    plt.close()

if __name__ == "__main__":
    # Generate both charts
    generate_comparative_chart()
    generate_rag_exclusive_chart()
    print("Charts generated successfully! You can now insert them into Chapter 4.")