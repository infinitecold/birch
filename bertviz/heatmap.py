from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

import attention, visualization


def get_args():
    parser = ArgumentParser(description="heatmap visualization for BERT attention")
    parser.add_argument("--model_path", default="models/saved.mb_1")
    parser.add_argument("--layer", default=0, type=int)  # 0 <= layer <= 23
    parser.add_argument("--head", default=0, type=int)  # 0 <= head <= 15
    parser.add_argument("--query", default="alternative medicine")
    parser.add_argument("--sentence", default="Traditionally, conventional doctors have been suspicious of unlicensed healers (and vice versa), but that is changing")
    parser.add_argument("--type", default="all", choices=["all", "ab", "ba", "a", "b"])
    args, _ = parser.parse_known_args()

    return args


def plot_attn(attn_data, labels_a, labels_b):
    fig = plt.figure()
    sns.heatmap(attn_data,
                cmap="Blues",
                linewidths=0.5,
                cbar=False,
                square=True,
                xticklabels=labels_b,
                yticklabels=labels_a) 
    fig.tight_layout()
    fig.savefig("attn.png", dpi=1000)


def plot_all_layers_attn(attn_data, labels_a, labels_b, args):
    if not os.path.isdir("attn"):
        os.mkdir("attn")

    for i in range(24):
        for j in range(16):
            fig = plt.figure()
            sns.heatmap(attn_data[i][args.head],
                        cmap="Blues",
                        linewidths=0.5,
                        cbar=False,
                        square=True,
                        xticklabels=labels_b,
                        yticklabels=labels_a)
            fig.tight_layout()
            fig.savefig("attn/attn-layer" + str(i).zfill(2) + "head" + str(j).zfill(2) + ".png", dpi=500)
            plt.close(fig)


if __name__ == "__main__":
    args = get_args()

    viz = visualization.AttentionVisualizer(args.model_path)
    tokens_a, tokens_b, attn = viz.get_viz_data(args.query, args.sentence)
    plot_all_layers_attn(attention.get(tokens_a, tokens_b, attn)[args.type]["att"], tokens_a + tokens_b, tokens_a + tokens_b, args)
    exit(0)
    
    attn_data = attention.get(tokens_a, tokens_b, attn)[args.type]["att"][args.layer][args.head]

    if args.type == "all":
        plot_attn(attn_data, tokens_a + tokens_b, tokens_a + tokens_b)
    elif args.type == "ab":
        plot_attn(attn_data, tokens_a, tokens_b)
    elif args.type == "ba":
        plot_attn(attn_data, tokens_b, tokens_a)
    elif args.type == "a":
        plot_attn(attn_data, tokens_a, tokens_a)
    elif args.type == "b":
        plot_attn(attn_data, tokens_b, tokens_b)
