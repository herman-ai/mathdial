import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="white")

def main():
    plot_dimension = sys.argv[1]

    if plot_dimension not in ['self-typical-confusion', 'self-typical-interactions']:
        print("Invalid plot dimension. Please choose 'self-typical-confusion' or 'self-typical-interactions'.")
        return

    with open('../data/train.jsonl', 'r') as f:
        train_data = [json.loads(line) for line in f]
        if plot_dimension == 'self-typical-confusion':
            confusion_train_data = [example['self-typical-confusion'] for example in train_data if 'self-typical-confusion' in example and example['self-typical-confusion'] is not None]
        else:
            typical_train_data = [example['self-typical-interactions'] for example in train_data if 'self-typical-interactions' in example and example['self-typical-interactions'] is not None]    

    with open('../data/test.jsonl', 'r') as f:
        test_data = [json.loads(line) for line in f]
        if plot_dimension == 'self-typical-confusion':
            confusion_test_data = [example['self-typical-confusion'] for example in test_data if 'self-typical-confusion' in example and example['self-typical-confusion'] is not None]
        else:
            typical_test_data = [example['self-typical-interactions'] for example in test_data if 'self-typical-interactions' in example and example['self-typical-interactions'] is not None]

    if plot_dimension == 'self-typical-confusion':
        df = pd.DataFrame({
            'Self-Typical Confusion Score': np.concatenate([confusion_train_data, confusion_test_data]),
            'Dataset': ['Train'] * len(confusion_train_data) + ['Val'] * len(confusion_test_data)
        })

        bin_edges = np.arange(0.5, 6, 1)
        # Compute normalized histograms for each dataset
        train_counts, _ = np.histogram(confusion_train_data, bins=bin_edges)
        test_counts, _ = np.histogram(confusion_test_data, bins=bin_edges)
        train_probs = train_counts / train_counts.sum()
        test_probs = test_counts / test_counts.sum()

        # Plot as bar chart for probability distributions
        width = 0.35
        x = np.arange(1, 6)
        plt.bar(x - width/2, train_probs, width, label=f'Train (n={len(confusion_train_data)})', color='tab:blue', edgecolor='black')
        plt.bar(x + width/2, test_probs, width, label=f'Val (n={len(confusion_test_data)})', color='tab:orange', edgecolor='black')
        plt.xlabel('Self-Typical Confusion Score')
        plt.ylabel('Probability')
        plt.xticks([1, 2, 3, 4, 5])
        plt.title('Distribution of Self-Typical Confusion Scores')
        plt.legend()
        plt.grid(axis='x', alpha=1)
        plt.savefig('self_typical_confusion_distribution.png')
    # plt.show()
    else:
        bin_edges = np.arange(0.5, 6, 1)
        df = pd.DataFrame({
            'Self-Typical Interaction score': np.concatenate([typical_train_data, typical_test_data]),
            'Dataset': ['Train'] * len(typical_train_data) + ['Val'] * len(typical_test_data)
        })

        # Compute normalized histograms for each dataset
        train_counts, _ = np.histogram(typical_train_data, bins=bin_edges)
        test_counts, _ = np.histogram(typical_test_data, bins=bin_edges)
        train_probs = train_counts / train_counts.sum()
        test_probs = test_counts / test_counts.sum()

        # Plot as bar chart for probability distributions
        width = 0.35
        x = np.arange(1, 6)
        plt.bar(x - width/2, train_probs, width, label=f'Train (n={len(typical_train_data)})', color='tab:blue', edgecolor='black')
        plt.bar(x + width/2, test_probs, width, label=f'Val (n={len(typical_test_data)})', color='tab:orange', edgecolor='black')
        plt.xlabel('Self-Typical Interaction score')
        plt.ylabel('Probability')
        plt.xticks([1, 2, 3, 4, 5])
        plt.title('Distribution of Self-Typical Interaction Scores')
        plt.legend()
        plt.grid(axis='y', alpha=1)
        plt.savefig('self_typical_interactions_distribution.png')


if __name__ == "__main__":
    main()