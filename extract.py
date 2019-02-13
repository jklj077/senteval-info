import collections

import numpy as np


def format(results):
    # header: STS12 p, STS12 s, STS13 p, STS13 s, STS14 p, STS14 s, STS15 p, STS15 s, STS 16 p, STS16 s, STS B p, STS B s, STS B m, SICK-R p, SICK-R s, SICK-P m
    line = ""
    if "STS12" in results:
        line += "{:.4f}".format(results["STS12"]["all"]["pearson"]["wmean"])
        line += "/"
        line += "{:.4f}".format(results["STS12"]["all"]["spearman"]["wmean"])
        line += "|"
    else:
        line += "na/na|"

    if "STS13" in results:
        line += "{:.4f}".format(results["STS13"]["all"]["pearson"]["wmean"])
        line += "/"
        line += "{:.4f}".format(results["STS13"]["all"]["spearman"]["wmean"])
        line += "|"
    else:
        line += "na/na|"

    if "STS14" in results:
        line += "{:.4f}".format(results["STS14"]["all"]["pearson"]["wmean"])
        line += "/"
        line += "{:.4f}".format(results["STS14"]["all"]["spearman"]["wmean"])
        line += "|"
    else:
        line += "na/na|"

    if "STS15" in results:
        line += "{:.4f}".format(results["STS15"]["all"]["pearson"]["wmean"])
        line += "/"
        line += "{:.4f}".format(results["STS15"]["all"]["spearman"]["wmean"])
        line += "|"
    else:
        line += "na/na|"

    if "STS16" in results:
        line += "{:.4f}".format(results["STS16"]["all"]["pearson"]["wmean"])
        line += "/"
        line += "{:.4f}".format(results["STS16"]["all"]["spearman"]["wmean"])
        line += "|"
    else:
        line += "na/na|"

    if "STSBenchmark" in results:
        line += "{:.4f}".format(results["STSBenchmark"]["pearson"])
        line += "/"
        line += "{:.4f}".format(results["STSBenchmark"]["spearman"])
        line += "/"
        line += "{:.4f}".format(results["STSBenchmark"]["mse"])
        line += "|"
    else:
        line += "na/na/na|"

    if "SICKRelatedness" in results:
        line += "{:.4f}".format(results["SICKRelatedness"]["pearson"])
        line += "/"
        line += "{:.4f}".format(results["SICKRelatedness"]["spearman"])
        line += "/"
        line += "{:.4f}".format(results["SICKRelatedness"]["mse"])
        line += "|"
    else:
        line += "na/na/na|"

    print(line)

    # header: MR, CR, SUBJ, MPQA, SST-B, SST-F, TREC, SICK-E, SNLI, MRPC, MRPC f

    line = ""
    for task in [
        "MR",
        "CR",
        "SUBJ",
        "MPQA",
        "SST2",
        "SST5",
        "TREC",
        "SICKEntailment",
        "SNLI",
    ]:
        if task in results:
            line += "{:.2f}".format(results[task]["acc"])
            line += "|"
        else:
            line += "na|"

    if "MRPC" in results:
        line += "{:.2f}".format(results["MRPC"]["acc"])
        line += "/"
        line += "{:.2f}".format(results["MRPC"]["f1"])
        line += "|"
    else:
        line += "na/na|"

    print(line)

    # header: COCO r1i2t, COCO r5i2t, COCO r10i2t, COCO medr_i2t, COCO r1t2i, COCO r5t2i, COCO r10t2i, COCO medr_t2i
    if "ImageCaptionRetrieval" in results:
        values = list(results["ImageCaptionRetrieval"]["acc"][0]) + list(
            results["ImageCaptionRetrieval"]["acc"][1]
        )
        values = ["{:.2f}".format(v) for v in values]
    else:
        values = ["na"] * 10

    print("|".join(values))

    # header: SentLen, WC, TreeDepth, TopConst, BShift, Tense, SubjNum, ObjNum, SOMO, CoordInv, average
    values = []
    for task in [
        "Length",
        "WordContent",
        "Depth",
        "TopConstituents",
        "BigramShift",
        "Tense",
        "SubjNumber",
        "ObjNumber",
        "OddManOut",
        "CoordinationInversion",
    ]:
        if task in results:
            values.append(results[task]["acc"])
        else:
            values.append("na")

    if "na" in values:
        values.append("na")
    else:
        values.append(sum(values) / len(values))

    values = ["{:.2f}".format(v) if v != "na" else v for v in values]

    line = "|".join(values)
    print(line)


if __name__ == "__main__":
    SpearmanrResult = collections.namedtuple(
        "SpearmanrResult", "correlation, pvalue"
    )
    array = np.array

    with open("temp.txt", "r", encoding="utf8") as f:
        c = f.read()
        format(eval(c))
