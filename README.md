# SentEval

Using SentEval to evaluate the word vectors and hidden vectors of pretrained models for natural language processing.

To aggregate the hidden vectors of a sentence to a single vector, three methods are used: using the vector at the first position ([CLS] symbol), using the vector at the last position ([SEP] symbol), using the average across the sentence, and using the maximum across the sentence.

## Bert-base-uncased

There are 12 layers in the bert-base-uncased model, denoted as 1-12. The word vectors are also included, denoted as 0. Modified codes from huggingface and pretrained model from goolge are used. The colors in the heatmaps are normalized to show the trend across layers.

### Semantic Relatedness


SkipThought-LN < GloVe BoW < fastText Bow < InferSent < Char-phrase

#### first

Some NaNs are encountered at the word embedding layer (white cells). L4 or L5 are the best performing layers, but are still not favorable. 

![bert-base-uncased-first-relatedness](pics/bert-base-uncased-first-relatedness.png)

 

#### last

Results are better than _first_. Best performing layers spread from L4 to L8. Notice that the last layer (L12) performs worse than the last layer of _first_.

![bert-base-uncased-first-relatedness](pics/bert-base-uncased-last-relatedness.png)



#### max

Good performance for L0, L1, and L2. They are the bottom layers.

![bert-base-uncased-first-relatedness](pics/bert-base-uncased-max-relatedness.png)

#### mean

Good performance for L1 and L2, better than _max_. In general, better than fasttext BoW, on par with InferSent. They are almost the bottom layers.

![bert-base-uncased-first-relatedness](pics/bert-base-uncased-mean-relatedness.png)

### Text Classification

GloVe < fastText < SkipThought < InferSent

#### first

The higher, the better. Good performance, almost as good as SkipThought. Maybe the next sentece prediction task is essential for BERT?

![bert-base-uncased-first-classification](pics/bert-base-uncased-first-classification.png)

#### last

The higher, the better. Not as good as _first_.

![bert-base-uncased-first-classification](pics/bert-base-uncased-last-classification.png)

#### max

The higher, the better. Not as good as _last_.

![bert-base-uncased-first-classification](pics/bert-base-uncased-max-classification.png)

#### mean

The higher, the better. Better than _first_, arguably better than SkipThought.

![bert-base-uncased-first-classification](pics/bert-base-uncased-mean-classification.png)

### Probing Tasks

fastText BoW < NLI < SkipThought (except that SkipThought is realy bad at WC) < AutoEncoder < NMT < Seq2Tree. These reflect linguistic properties.

#### first

Not good. 

![bert-base-uncased-first-probing](pics/bert-base-uncased-first-probing.png)

#### last

Worse.

![bert-base-uncased-first-probing](pics/bert-base-uncased-last-probing.png)

#### max

Not good.

![bert-base-uncased-first-probing](pics/bert-base-uncased-max-probing.png)

#### mean

Best performing. But still lag far behind NMT or Seq2Tree, on par with NLI pretraining of BiLSTM and GatedConvNet.

![bert-base-uncased-first-probing](pics/bert-base-uncased-mean-probing.png)



### Discussion/Hypothesis

- BERT's downstream tasks are all classification tasks, so maybe it is not surprising to see its hidden layers without task specific fine-tuning is also very good at classification.
- BERT's [CLS] vector and [SEP] vector embody a lot of information, maybe too much information. It may root from the next sentence prediction task, which makes use of the two vectors and which makes SkipThought succesful at classification task as well. (How does masked language modelling function here?)
- BERT's hidden representation is not as good in Semantic Relatedness or Linguistic Properties as in Text Classification, similar to SkipThought or NLI pretraining. These aspects seem unimportant for classification tasks or extraction tasks; however, they may be important for generation.
- From early observation on BERT attention distribution, it is also found that only the attention of the bottom layers are spread over the whole sentence and in higher layers, the tokens most attend to [CLS] and [SEP] symbols. It is possible that only the bottom layers are gathering and combining semantics, as shown by the semantic relatedness results. [CLS] and [SEP] become actual sentence representations in higher layers. Task-specific fine-tuning may change that, but how much? (BERT-base-uncased seems overfitting too fast on SQuAD fine-tuning.)

TODO: evaluate bert-large-uncased

