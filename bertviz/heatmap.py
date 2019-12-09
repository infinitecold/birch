from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os
import seaborn as sns

import attention, visualization


def get_args():
    parser = ArgumentParser(description="Heatmap visualization for BERT attention")
    parser.add_argument("--model_path", default="models/saved.mb_1")
    parser.add_argument("--query", default="alternative medicine")
    parser.add_argument("--sentence", default="Traditionally, conventional doctors have been suspicious of unlicensed healers (and vice versa), but that is changing")
    parser.add_argument("--type", default="all", choices=["all", "ab", "ba", "a", "b"])
    parser.add_argument("--plot_all", action="store_true")
    parser.add_argument("--layer", default=0, type=int, choices=range(24))
    parser.add_argument("--head", default=0, type=int, choices=range(16))
    args, _ = parser.parse_known_args()

    return args


def plot_attn(attn_data, labels_a, labels_b, output_fp):
    fig = plt.figure()
    sns.heatmap(attn_data,
                cmap="Blues",
                linewidths=0.5,
                cbar=False,
                square=True,
                xticklabels=labels_b,
                yticklabels=labels_a) 
    fig.tight_layout()
    fig.savefig(output_fp, dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    args = get_args()

    # make directories to store heatmaps
    if not os.path.isdir("heatmaps"):
        os.mkdir("heatmaps")
    model_name = args.model_path[args.model_path.rfind("/")+1:]
    if not os.path.isdir(f"heatmaps/{model_name}"):
        os.mkdir(f"heatmaps/{model_name}")

    # get attention data from model
    viz = visualization.AttentionVisualizer(args.model_path)
    tokens_a, tokens_b, attn = viz.get_viz_data(args.query, args.sentence)

    # handle different attention types
    attn_data = attention.get(tokens_a, tokens_b, attn)[args.type]["att"]
    if args.type == "all":
        labels_a = tokens_a + tokens_b
        labels_b = tokens_a + tokens_b
    elif args.type == "ab":
        labels_a = tokens_a
        labels_b = tokens_b
    elif args.type == "ba":
        labels_a = tokens_b
        labels_b = tokens_a
    elif args.type == "a":
        labels_a = tokens_a
        labels_b = tokens_a
    elif args.type == "b":
        labels_a = tokens_b
        labels_b = tokens_b

    # plot heatmaps
    if args.plot_all:
        for i in range(24):
            for j in range(16):
                output_fp = f"heatmaps/{model_name}/attn-layer{str(i).zfill(2)}head{str(j).zfill(2)}.png"
                plot_attn(attn_data[i][j], labels_a, labels_b, output_fp)
    else:
        output_fp = f"heatmaps/{model_name}/attn-layer{str(args.layer).zfill(2)}head{str(args.head).zfill(2)}.png"
        plot_attn(attn_data[args.layer][args.head], labels_a, labels_b, output_fp)
