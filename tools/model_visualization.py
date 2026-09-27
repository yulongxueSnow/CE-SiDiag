import matplotlib.pyplot as plt
import seaborn as sns

def visualize_chaos_features(X, chaos_features, labels):

    # Original feature distribution and chaos feature distribution,
    # selecting 2 features, shown as scatterplots in one figure with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Original feature distribution
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis', ax=axs[0])
    axs[0].set_title('Original Features Distribution')
    axs[0].set_xlabel('Feature 1')
    axs[0].set_ylabel('Feature 2')
    axs[0].legend(title='Label')

    # Chaos feature distribution
    sns.scatterplot(x=chaos_features[:, 0], y=chaos_features[:, 1], hue=labels, palette='viridis', ax=axs[1])
    axs[1].set_title('Chaos Features Distribution')
    axs[1].set_xlabel('Feature 1')
    axs[1].set_ylabel('Feature 2')
    axs[1].legend(title='Label')

    plt.tight_layout()  # Adjust subplot spacing
    plt.show()

    
    