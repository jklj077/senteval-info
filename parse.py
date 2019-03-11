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

    if verbose:
        print(labels)
        print(values)

    normalized_values = (values - np.nanmin(values, axis=0)[np.newaxis, :]) / (
        np.nanmax(values, axis=0) - np.nanmin(values, axis=0)
    )[np.newaxis, :]

    fig, ax = plt.subplots()
    im = ax.imshow(normalized_values)

    for i in range(len(layers)):
        for j in range(len(labels)):
            text = ax.text(
                j, i, values[i, j], ha="center", va="center", color="w"
            )

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(layers)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(layers)

    ax.set_xlabel("tasks")
    ax.set_ylabel("layers")

    ax.xaxis.tick_top()
    ax.invert_yaxis()

    ax.set_title(title)

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

    draw_heatmap(relatedness_labels, relatedness_values, file, verbose)
    draw_heatmap(classification_labels, classification_values, file, verbose)
    draw_heatmap(coco_labels, coco_values, file, verbose)
    draw_heatmap(probe_labels, probe_values, file, verbose)


if __name__ == "__main__":
    parse("logs/log-first-bert-base-uncased.txt", True)
    parse("logs/log-last-bert-base-uncased.txt", True)
    parse("logs/log-max-bert-base-uncased.txt", True)
    parse("logs/log-mean-bert-base-uncased.txt", True)

