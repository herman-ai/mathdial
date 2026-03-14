import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="white")

def parse_conversation(conversation_str):
    '''Parse the conversation string into a list of (speaker, turn_text) tuples.

    conversationample:
    [
        ("Teacher","Hi ..."),
        ("Student","I got 72 ..."),
        ("Teacher","...")
    ]
    
    '''
    turns = []
    for turn in conversation_str.split("|EOM|"):
        turn = turn.strip() # remove whitespace
        if not turn or ": " not in turn:
            print(f"Improper formatting, skipping turn. Should not happen: '{turn}'")
            continue
        speaker, turn_text = turn.split(": ", 1) # split string into these two parts to format into tuple
        turns.append((speaker.strip(), turn_text.strip()))
    return turns

def parse_conversation_length(conversation_str):
    turns = [t.strip() for t in conversation_str.split('|EOM|') if t.strip()]
    return len(turns)

def main():
    plot_dimension = sys.argv[1]

    if plot_dimension not in ['self-typical-confusion', 'self-typical-interactions', 'self-correctness', 'conversation-length']:
        print("Invalid plot dimension. Please choose 'self-typical-confusion', 'self-typical-interactions', 'self-correctness', or 'conversation-length'.")
        return

    with open('../data/train.jsonl', 'r') as f:
        train_data = [json.loads(line) for line in f]
        if plot_dimension == 'self-typical-confusion':
            confusion_train_data = [example['self-typical-confusion'] for example in train_data if 'self-typical-confusion' in example and example['self-typical-confusion'] is not None]
        elif plot_dimension == 'self-typical-interactions':
            typical_train_data = [example['self-typical-interactions'] for example in train_data if 'self-typical-interactions' in example and example['self-typical-interactions'] is not None]
        elif plot_dimension == 'self-correctness':
            correctness_train_data = [example['self-correctness'] for example in train_data if 'self-correctness' in example and example['self-correctness'] is not None]
        else:
            conversation_length_train_data = [parse_conversation_length(example['conversation']) for example in train_data if 'conversation' in example and example['conversation'] is not None]

    with open('../data/test.jsonl', 'r') as f:
        test_data = [json.loads(line) for line in f]
        if plot_dimension == 'self-typical-confusion':
            confusion_test_data = [example['self-typical-confusion'] for example in test_data if 'self-typical-confusion' in example and example['self-typical-confusion'] is not None]
        elif plot_dimension == 'self-typical-interactions':
            typical_test_data = [example['self-typical-interactions'] for example in test_data if 'self-typical-interactions' in example and example['self-typical-interactions'] is not None]
        elif plot_dimension == 'self-correctness':
            correctness_test_data = [example['self-correctness'] for example in test_data if 'self-correctness' in example and example['self-correctness'] is not None]
        else:
            conversation_length_test_data = [parse_conversation_length(example['conversation']) for example in test_data if 'conversation' in example and example['conversation'] is not None]

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
        # Only show left and bottom spines (borders)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        plt.savefig('plots/dataset_stats/self_typical_confusion_distribution.png')
    # plt.show()
    elif plot_dimension == 'self-typical-interactions':
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
        # Only show left and bottom spines (borders)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        plt.savefig('plots/dataset_stats/self_typical_interactions_distribution.png')
    elif plot_dimension == 'self-correctness':
        # For self-correctness, count 'Yes' and 'No' and plot as bar chart
        train_counts = pd.Series(correctness_train_data).value_counts().sort_index()
        test_counts = pd.Series(correctness_test_data).value_counts().sort_index()
        categories = sorted(set(train_counts.index).union(set(test_counts.index)))
        train_probs = train_counts.reindex(categories, fill_value=0) / train_counts.sum()
        test_probs = test_counts.reindex(categories, fill_value=0) / test_counts.sum()
        width = 0.35
        x = np.arange(len(categories))
        plt.bar(x - width/2, train_probs, width, label=f'Train (n={len(correctness_train_data)})', color='tab:blue', edgecolor='black')
        plt.bar(x + width/2, test_probs, width, label=f'Val (n={len(correctness_test_data)})', color='tab:orange', edgecolor='black')
        plt.xlabel('Self-Correctness')
        plt.ylabel('Probability')
        plt.xticks(x, categories)
        plt.title('Distribution of Self-Correctness')
        plt.legend()
        plt.grid(axis='y', alpha=1)
        # Only show left and bottom spines (borders)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        plt.savefig('plots/dataset_stats/self_correctness_distribution.png')
    else:
        # For conversation length, plot as histogram
        print(np.array(conversation_length_train_data).shape, np.array(conversation_length_test_data).shape)
        df = pd.DataFrame({
            'Conversation Length': np.concatenate([conversation_length_train_data, conversation_length_test_data]),
            'Dataset': ['Train'] * len(conversation_length_train_data) + ['Val'] * len(conversation_length_test_data)
        })
        # Plot as bar chart for probability distributions (not density)
        plt.figure(figsize=(10, 6))
        train_counts, bins = np.histogram(conversation_length_train_data, bins=20)
        test_counts, _ = np.histogram(conversation_length_test_data, bins=bins)
        train_probs = train_counts / train_counts.sum()
        test_probs = test_counts / test_counts.sum()
        bin_centers = (bins[:-1] + bins[1:]) / 2
        width = (bins[1] - bins[0]) * 0.4
        colors = plt.cm.tab10(np.linspace(0, 0.9, 2))
        plt.bar(bin_centers - width/2, train_probs, width=width, label=f'Train (n={len(conversation_length_train_data)})', color=colors[0], edgecolor="white", linewidth=0.8)
        plt.bar(bin_centers + width/2, test_probs, width=width, label=f'Val (n={len(conversation_length_test_data)})', color=colors[1], edgecolor="white", linewidth=0.8)
        plt.xlabel('Conversation Length (number of turns)', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylabel('Probability', fontsize=16, fontweight='bold')
        plt.title('Distribution of Conversation Lengths', fontsize=16, fontweight='bold', pad=12)
        plt.legend(title='Dataset', fontsize=16)
        plt.tight_layout()
        plt.grid(axis='y', alpha=1, linestyle='--')
        # Only show left and bottom spines (borders)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        plt.savefig('plots/conversation_stats/conversation_length_distribution.png')


if __name__ == "__main__":
    main()