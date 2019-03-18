import matplotlib.pyplot as plt
import numpy as np

relatedness_labels = "STS12 p, STS12 s, STS13 p, STS13 s, STS14 p, STS14 s, STS15 p, STS15 s, STS 16 p, STS16 s, STS B p, STS B s, STS B m, SICK-R p, SICK-R s, SICK-P m"
classification_labels = (
    "MR, CR, SUBJ, MPQA, SST-B, SST-F, TREC, SICK-E, SNLI, MRPC, MRPC f"
)
coco_labels = "COCO r1i2t, COCO r5i2t, COCO r10i2t, COCO medr_i2t, COCO r1t2i, COCO r5t2i, COCO r10t2i, COCO medr_t2i"
probe_labels = "SentLen, WC, TreeDepth, TopConst, BShift, Tense, SubjNum, ObjNum, SOMO, CoordInv, average"


def draw_heatmap(labels, values, title, verbose=False):

    labels = labels.split(", ")
    layers = np.arange(len(values), dtype=np.int)
    values = np.row_stack(np.fromstring(line, sep=",") for line in values)

    if "STS B m" in labels:
        labels = [x for x in labels if " m" not in x]
        values = np.delete(values, [12, 15], axis=1)
        precision = 4
    else:
        precision = 2

    if not "average" in labels:
        labels.append("average")
        values = np.column_stack(
            [values, np.around(np.nanmean(values, axis=1), precision)]
        )

    if verbose:
        print(labels)
        print(values)

    normalized_values = (values - np.nanmin(values, axis=0)[np.newaxis, :]) / (
        np.nanmax(values, axis=0) - np.nanmin(values, axis=0)
    )[np.newaxis, :]

    fig, ax1 = plt.subplots()

    im = ax1.imshow(
        normalized_values, aspect="auto", cmap=plt.get_cmap("Blues")
    )

    for i in range(len(layers)):
        for j in range(len(labels)):
            color = "black" if normalized_values[i, j] < 0.5 else "white"
            text = ax1.text(
                j, i, values[i, j], ha="center", va="center", color=color
            )

    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_yticks(np.arange(len(layers)))
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(layers)

    ax1.set_xlabel("tasks")
    ax1.set_ylabel("layers")

    ax1.xaxis.tick_top()
    ax1.invert_yaxis()

    ax1.set_title(title)

    plt.show()


def draw_heatmap(labels, values, title, verbose=False, save_name=None):

    labels = labels.split(", ")
    layers = np.arange(len(values), dtype=np.int)
    values = np.row_stack(np.fromstring(line, sep=",") for line in values)

    if "STS B m" in labels:
        labels = [x for x in labels if " m" not in x]
        values = np.delete(values, [12, 15], axis=1)
        precision = 4
    else:
        precision = 2

    if not "average" in labels:
        labels.append("average")
        values = np.column_stack(
            [values, np.around(np.nanmean(values, axis=1), precision)]
        )

    if verbose:
        print(labels)
        print(values)

    normalized_values = (values - np.nanmin(values, axis=0)[np.newaxis, :]) / (
        np.nanmax(values, axis=0) - np.nanmin(values, axis=0)
    )[np.newaxis, :]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        gridspec_kw={"width_ratios": [len(labels) - 1, 1]},
        figsize=(19.34375, 10.27083),
        # dpi=96,
        constrained_layout=True,
    )
    # fig.subplots_adjust(wspace=0.045)

    im = ax1.imshow(
        normalized_values[:, :-1], aspect="auto", cmap=plt.get_cmap("Blues")
    )

    for i in range(len(layers)):
        for j in range(len(labels) - 1):
            color = "black" if normalized_values[i, j] < 0.5 else "white"
            text = ax1.text(
                j, i, values[i, j], ha="center", va="center", color=color
            )

    ax1.set_xticks(np.arange(len(labels) - 1))
    ax1.set_yticks(np.arange(len(layers)))
    ax1.set_xticklabels(labels[:-1])
    ax1.set_yticklabels(layers)

    ax1.set_xlabel("tasks")
    ax1.set_ylabel("layers")

    ax1.xaxis.tick_top()
    ax1.invert_yaxis()

    im = ax2.imshow(
        normalized_values[:, -1][:, np.newaxis],
        aspect="auto",
        cmap=plt.get_cmap("Blues"),
    )

    for i in range(len(layers)):
        # for j in range(len(labels)-1):
        color = "black" if normalized_values[i, -1] < 0.5 else "white"
        text = ax2.text(
            0, i, values[i, -1], ha="center", va="center", color=color
        )

    ax2.set_xticks([0])
    ax2.set_yticks(np.arange(len(layers)))
    ax2.set_xticklabels([labels[-1]])
    ax2.set_yticklabels(layers)

    # ax2.set_xlabel("tasks")
    ax2.set_ylabel("layers")

    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.invert_yaxis()

    # fig.suptitle(title)

    ax1.set_title(title)
    # fig.tight_layout()
    if save_name is not None:
        plt.savefig(save_name, dpi=96, transparent=True)
    else:
        plt.show()


def parse(file, verbose=False):

    relatedness_values = []
    classification_values = []
    coco_values = []
    probe_values = []

    with open(file, "r", encoding="utf8") as fin:
        while True:
            line = fin.readline()
            if not line:
                break

            if "total results: " not in line:
                continue

            fin.readline()  # key=value
            fin.readline()  # key
            line = fin.readline()  # value
            relatedness_values.append(line.strip().split(":")[-1])

            fin.readline()  # key=value
            fin.readline()  # key
            line = fin.readline()  # value
            classification_values.append(line.strip().split(":")[-1])

            fin.readline()  # key=value
            fin.readline()  # key
            line = fin.readline()  # value
            coco_values.append(line.strip().split(":")[-1])

            fin.readline()  # key=value
            fin.readline()  # key
            line = fin.readline()  # value
            probe_values.append(line.strip().split(":")[-1])

    parts = file[9:-4].split("-")
    parts = parts[1:] + [parts[0]]
    prefix = "-".join(parts)
    draw_heatmap(
        relatedness_labels,
        relatedness_values,
        file,
        verbose,
        save_name="{}-{}.svg".format(prefix, "relatedness"),
    )
    draw_heatmap(
        classification_labels,
        classification_values,
        file,
        verbose,
        save_name="{}-{}.svg".format(prefix, "classification"),
    )
    # draw_heatmap(coco_labels, coco_values, file, verbose)
    draw_heatmap(
        probe_labels,
        probe_values,
        file,
        verbose,
        save_name="{}-{}.svg".format(prefix, "probing"),
    )


if __name__ == "__main__":
    parse("logs/log-first-bert-base-uncased.txt", True)
    parse("logs/log-last-bert-base-uncased.txt", True)
    parse("logs/log-max-bert-base-uncased.txt", True)
    parse("logs/log-mean-bert-base-uncased.txt", True)

    parse("logs/log-first-bert-large-uncased.txt", True)
    parse("logs/log-last-bert-large-uncased.txt", True)
    parse("logs/log-max-bert-large-uncased.txt", True)
    parse("logs/log-mean-bert-large-uncased.txt", True)
