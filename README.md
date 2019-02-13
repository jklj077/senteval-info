# SentEval

Using SentEval to evaluate the word vectors and hidden vectors of pretrained models for natural language processing.

To aggregate the hidden vectors of a sentence to a single vector, three methods are used: using the vector at the first position, using the average across the sentence, and using the maximum across the sentence.

## Bert-base-uncased

There are 12 layers in the bert-base-uncased model, denoted as 1-12. The word vectors are also included, denoted as 0.

### Semantic textual similarity

#### first

| Layer | STS12         | STS13         | STS14         | STS15         | STS16         | STS B                | SICK-R               |
| ----- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------------- | -------------------- |
| 0     | nan           | nan           | nan           | nan           | nan           | 0/0/2.47             | 0/0/1.02             |
| 1     | 0.1588/0.2148 | 0.0802/0.1507 | 0.1027/0.1904 | 0.1207/0.2486 | 0.1070/0.3009 | 0.49/0.48/2.00       | 0.72/0.67/0.49       |
| 2     | 0.2528/0.2865 | 0.1183/0.1928 | 0.1720/0.2469 | 0.1924/0.3205 | 0.2312/0.3795 | 0.53/0.52/1.98       | 0.73/0.68/0.17       |
| 3     | 0.2415/0.2951 | 0.1301/0.2172 | 0.1511/0.2486 | 0.1533/0.3165 | 0.2024/0.4121 | 0.5413/0.5348/1.9346 | 0.7527/0.6884/0.4449 |
| 4     | 0.3851/0.4340 | 0.2953/0.3733 | 0.3721/0.4165 | 0.3249/0.4698 | 0.4394/0.5900 | 0.6425/0.6385/1.6652 | 0.7722/0.6981/0.4109 |
| 5     | 0.3716/0.3977 | 0.3282/0.3517 | 0.3574/0.3640 | 0.3770/0.4498 | 0.5118/0.5899 | 0.6203/0.6151/1.6715 | 0.7556/0.6920/0.4419 |
| 6     | 0.3096/0.3194 | 0.3093/0.3105 | 0.3109/0.3103 | 0.3361/0.3602 | 0.4634/0.5306 | 0.5122/0.5075/1.8177 | 0.7210/0.6577/0.4989 |
| 7     | 0.2754/0.2926 | 0.2199/0.2220 | 0.2761/0.2702 | 0.2977/0.3218 | 0.4439/0.5012 | 0.4656/0.4629/1.9249 | 0.6971/0.6422/0.5263 |
| 8     |               |               |               |               |               |                      |                      |
| 9     |               |               |               |               |               |                      |                      |
| 10    |               |               |               |               |               |                      |                      |
| 11    |               |               |               |               |               |                      |                      |
| 12    |               |               |               |               |               |                      |                      |

### Classification

#### first

| Layer | MR    | CR    | SUBJ  | MPQA  | SST-B | SST-F | TREC | SICK-E | SNLI | MRPC        | COCO |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ---- | ------ | ---- | ----------- | ---- |
| 0     | 50    | 63.76 | 50    | 68.77 | 49.92 | 28.64 | 18.8 | 56.69  |      | 66.49/79.87 |      |
| 1     | 54.12 | 64.16 | 82.41 | 81.85 | 70.57 | 30.72 | 62.2 | 68.95  |      | 68.93/80.03 |      |
| 2     | 63.21 | 65.75 | 89.34 | 82.04 | 77.32 | 35.29 | 70.6 | 66.02  |      | 66.78/72.78 |      |
| 3     | 66.26 | 65.75 | 87.94 | 82.32 | 73.48 | 36.02 | 75.2 | 67.77  |      | 71.83/80.94 |      |
| 4     | 65.33 | 64.03 | 91.02 | 81.84 | 76.83 | 37.38 | 67.8 | 74.08  |      | 67.54/80.35 |      |
| 5     | 68.13 | 69.38 | 92.68 | 84.5  | 77.81 | 38.73 | 79.2 | 70.98  |      | 67.42/75.46 |      |
| 6     | 70.79 | 70.57 | 93.85 | 86.19 | 77.81 | 41.49 | 86.0 | 72.17  |      | 71.88/81.17 |      |
| 7     | 73.03 | 75.74 | 94.35 | 86.0  | 78.14 | 42.76 | 86.6 | 71.48  |      | 68.06/80.54 |      |
| 8     |       |       |       |       |       |       |      |        |      |             |      |
| 9     |       |       |       |       |       |       |      |        |      |             |      |
| 10    |       |       |       |       |       |       |      |        |      |             |      |
| 11    |       |       |       |       |       |       |      |        |      |             |      |
| 12    |       |       |       |       |       |       |      |        |      |             |      |

### Probing

#### first

| Layer | SentLen | WC   | TreeDepth | TopConst | BShift | Tense | SubjNum | ObjNum | SOMO | CoordInv |
| ----- | ------- | ---- | --------- | -------- | ------ | ----- | ------- | ------ | ---- | -------- |
| 0     | 16.7    | 0.1  | 17.9      | 5.0      | 50.0   | 50.0  | 50.0    | 50.0   | 49.9 | 50       |
| 1     | 71.9    | 0.8  | 26.1      | 51.0     | 50.0   | 78.2  | 73.4    | 71.9   | 49.9 | 51.8     |
| 2     | 84.2    | 2.2  | 30.9      | 52.6     | 50.8   | 83.7  | 79.3    | 77.7   | 51.2 | 51.1     |
| 3     | 87.2    | 1.9  | 30.2      | 48.0     | 51.9   | 85.5  | 79.0    | 77.4   | 53.9 | 53.8     |
| 4     | 85.7    | 11.4 | 29.4      | 49.7     | 63.9   | 86.5  | 76.0    | 78.2   | 55.1 | 50.4     |
| 5     | 75.0    | 10.5 | 29.8      | 68.3     | 81.1   | 88.7  | 84.3    | 79.2   | 57.9 | 54.6     |
| 6     | 75.3    | 15.1 | 29.3      | 73.0     | 80.6   | 89.0  | 88.4    | 78.4   | 56.7 | 64.8     |
| 7     | 68.9    | 23.0 | 30.0      | 76.7     | 81.2   | 89.3  | 87.6    | 79.5   | 57.6 | 68.6     |
| 8     |         |      |           |          |        |       |         |        |      |          |
| 9     |         |      |           |          |        |       |         |        |      |          |
| 10    |         |      |           |          |        |       |         |        |      |          |
| 11    |         |      |           |          |        |       |         |        |      |          |
| 12    |         |      |           |          |        |       |         |        |      |          |
