import matplotlib.pyplot as plt
import numpy as np


def plot_speaker_distr(index_df):
    # Mapping for POS codes to full names
    label_map = {'b': 'Beginning', 'm': 'Middle', 'e': 'End'}

    # Calculate POS totals
    pos_totals = index_df['pos'].value_counts()

    # Create display labels with totals
    pos_display_labels = {
        'b': f'Beginning ({pos_totals.get("b", 0)})',
        'm': f'Middle ({pos_totals.get("m", 0)})',
        'e': f'End ({pos_totals.get("e", 0)})'
    }

    # Create a new column with display labels for plotting
    df_display = index_df.copy()
    df_display['pos_display'] = df_display['pos'].map(pos_display_labels)

    # Group and pivot for stacked bar
    grouped = df_display.groupby(['speaker', 'pos_display']).size().unstack(fill_value=0)

    # Plot the stacked bar chart
    ax = grouped.plot(
        kind='bar',
        stacked=True,
        figsize=(10, 6),
        colormap='tab10'
    )

    # Axis labels and title
    ax.set_xlabel('Speaker ID')
    ax.set_ylabel('Number of Turns')

    # Legend inside plot at top right
    ax.legend(
        title='Position (Total)',
        loc='upper right',
        bbox_to_anchor=(1, 1),
        frameon=True
    )

    plt.tight_layout()
    plt.show()


# TODO: Make boxplots (once I have the data...)


def plot_results(results):

    positions = ['b', 'm', 'e']
    metrics = ['precision', 'recall', 'f1-score']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    pos_map = {"b": "Beginning", "m": "Middle", "e": "End"}
    cmap = plt.cm.tab10

    bar_width = 0.2
    x = np.arange(len(metrics))  # [0, 1, 2]

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, pos in enumerate(positions):
        means = [results[pos][0][metric].values[1] for metric in metrics]
        stdevs = [results[pos][1][metric].values[1] for metric in metrics]
        ci = 1.96 * np.array(stdevs)
        offset = (i - 1) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=ci, label=pos_map[pos],
            color=cmap(i), capsize=8, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylabel("Score")
    ax.set_title("Scores of RF Classifier")
    ax.set_ylim(0.5, 1.01)

    # Place legend outside the plot on the right
    ax.legend(title="Position", loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()
    return fig