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

    if plot_dimension not in ['self-typical-confusion', 'self-typical-interactions', 'self-correctness', 'conversation-length', 'combined-self-typical']:
        print("Invalid plot dimension. Please choose 'self-typical-confusion', 'self-typical-interactions', 'self-correctness', 'conversation-length', or 'combined-self-typical'.")
        return

    with open('../data/train.jsonl', 'r') as f:
        train_data = [json.loads(line) for line in f]
        if plot_dimension in ('self-typical-confusion', 'combined-self-typical'):
            confusion_train_data = [example['self-typical-confusion'] for example in train_data if 'self-typical-confusion' in example and example['self-typical-confusion'] is not None]
        if plot_dimension in ('self-typical-interactions', 'combined-self-typical'):
            typical_train_data = [example['self-typical-interactions'] for example in train_data if 'self-typical-interactions' in example and example['self-typical-interactions'] is not None]
        elif plot_dimension == 'self-correctness':
            correctness_train_data = [example['self-correctness'] for example in train_data if 'self-correctness' in example and example['self-correctness'] is not None]
        else:
            conversation_length_train_data = [parse_conversation_length(example['conversation']) for example in train_data if 'conversation' in example and example['conversation'] is not None]

    with open('../data/test.jsonl', 'r') as f:
        test_data = [json.loads(line) for line in f]
        if plot_dimension in ('self-typical-confusion', 'combined-self-typical'):
            confusion_test_data = [example['self-typical-confusion'] for example in test_data if 'self-typical-confusion' in example and example['self-typical-confusion'] is not None]
        if plot_dimension in ('self-typical-interactions', 'combined-self-typical'):
            typical_test_data = [example['self-typical-interactions'] for example in test_data if 'self-typical-interactions' in example and example['self-typical-interactions'] is not None]
        elif plot_dimension == 'self-correctness':
            correctness_test_data = [example['self-correctness'] for example in test_data if 'self-correctness' in example and example['self-correctness'] is not None]
        else:
            conversation_length_test_data = [parse_conversation_length(example['conversation']) for example in test_data if 'conversation' in example and example['conversation'] is not None]

    if plot_dimension == 'self-typical-confusion':
        bin_edges = np.arange(0.5, 6, 1)
        # Compute normalized histograms for each dataset
        train_counts, _ = np.histogram(confusion_train_data, bins=bin_edges)
        test_counts, _ = np.histogram(confusion_test_data, bins=bin_edges)
        train_probs = train_counts / train_counts.sum()
        test_probs = test_counts / test_counts.sum()

        # Plot as bar chart for probability distributions
        width = 0.35
        x = np.arange(1, 6)
        colors = plt.cm.tab10(np.linspace(0, 0.9, 2))
        plt.bar(x - width/2, train_probs, width, label=f'Train (n={len(confusion_train_data)})', color=colors[0], edgecolor='black')
        plt.bar(x + width/2, test_probs, width, label=f'Val (n={len(confusion_test_data)})', color=colors[1], edgecolor='black')
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
    elif plot_dimension == 'combined-self-typical':
        # Combined plot: Confusion (blues) and Interactions (greens), Train vs Val
        bin_edges = np.arange(0.5, 6, 1)
        x = np.arange(1, 6)
 
        # Compute normalized histograms
        conf_train_counts, _ = np.histogram(confusion_train_data, bins=bin_edges)
        conf_test_counts, _  = np.histogram(confusion_test_data,  bins=bin_edges)
        conf_train_probs = conf_train_counts / conf_train_counts.sum()
        conf_test_probs  = conf_test_counts  / conf_test_counts.sum()
 
        int_train_counts, _ = np.histogram(typical_train_data, bins=bin_edges)
        int_test_counts, _  = np.histogram(typical_test_data,  bins=bin_edges)
        int_train_probs = int_train_counts / int_train_counts.sum()
        int_test_probs  = int_test_counts  / int_test_counts.sum()
 
        # Four groups per x-tick: conf_train | conf_val | int_train | int_val
        width = 0.2
        offsets = [-1.5, -0.5, 0.5, 1.5]
 
        colors = plt.cm.tab10(np.linspace(0, 0.9, 10))  # expand to 10 to access all tab10 colors
        blue  = colors[0]  # same blue as your other plots
        green = colors[2]  # tab10 green, similar style

        # Then use alpha to distinguish Train vs Val:
        blue_dark=(*blue[:3],  1.0)   # Confusion Train    – full blue
        blue_light=(*blue[:3],  0.45)  # Confusion Val      – faded blue
        green_dark=(*green[:3], 1.0)   # Interactions Train – full green
        green_light=(*green[:3], 0.45)  # Interactions Val   – faded green
 
        fig, ax = plt.subplots(figsize=(10, 6))
 
        ax.bar(x + offsets[0] * width, conf_train_probs, width,
               label=f'Confusion – Train (n={len(confusion_train_data)})',
               color=blue_dark,  edgecolor='white', linewidth=0.6)
        ax.bar(x + offsets[1] * width, conf_test_probs,  width,
               label=f'Confusion – Val (n={len(confusion_test_data)})',
               color=blue_light, edgecolor='white', linewidth=0.6)
        ax.bar(x + offsets[2] * width, int_train_probs,  width,
               label=f'Interactions – Train (n={len(typical_train_data)})',
               color=green_dark,  edgecolor='white', linewidth=0.6)
        ax.bar(x + offsets[3] * width, int_test_probs,   width,
               label=f'Interactions – Val (n={len(typical_test_data)})',
               color=green_light, edgecolor='white', linewidth=0.6)
 
        ax.set_xlabel('Score', fontsize=16, fontweight='bold')
        ax.set_ylabel('Probability', fontsize=16, fontweight='bold')
        ax.set_title('Distribution of Self-Typical Confusion & Interaction Scores',
                     fontsize=16, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([1, 2, 3, 4, 5], fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.legend(fontsize=14, framealpha=0.9)
        ax.grid(axis='y', alpha=0.4, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
 
        plt.tight_layout()
        plt.savefig('plots/dataset_stats/combined_self_typical_distribution.png', dpi=150)
    elif plot_dimension == 'self-typical-interactions':
        bin_edges = np.arange(0.5, 6, 1)

        # Compute normalized histograms for each dataset
        train_counts, _ = np.histogram(typical_train_data, bins=bin_edges)
        test_counts, _ = np.histogram(typical_test_data, bins=bin_edges)
        train_probs = train_counts / train_counts.sum()
        test_probs = test_counts / test_counts.sum()

        # Plot as bar chart for probability distributions
        width = 0.35
        x = np.arange(1, 6)
        colors = plt.cm.tab10(np.linspace(0, 0.9, 2))
        plt.bar(x - width/2, train_probs, width, label=f'Train (n={len(typical_train_data)})', color=colors[0], edgecolor='black')
        plt.bar(x + width/2, test_probs, width, label=f'Val (n={len(typical_test_data)})', color=colors[1], edgecolor='black')
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
        colors = plt.cm.tab10(np.linspace(0, 0.9, 2))
        plt.bar(x - width/2, train_probs, width, label=f'Train (n={len(correctness_train_data)})', color=colors[0], edgecolor='black')
        plt.bar(x + width/2, test_probs, width, label=f'Val (n={len(correctness_test_data)})', color=colors[1], edgecolor='black')
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