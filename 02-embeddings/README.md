
Embeddings - SkipGram
=====================

The software in here tries to learn how to represent words as small-is vector
by learning their meaning from context using the Skip-Gram model. See [here][1]
and [here][2] for more details. It ends up working fairly well when trained on
the [sanitized wikipedia text][3] - [text8][4].

TensorBoard does a very nice job visualizing the embedding. You can query for
similarities and the T-SNE display is not entirely horrible even for such a big
number of classes:

![Embedding visualization](assets/embedding.png)

Also, the weight distribution over time in the embedding looks interesting:

![Weight distribution](assets/distribution.png)

FastText utils
--------------

`fasttext2npy.py` and `extract-embedding.py` are two utilities for manipulating
Facebook's [FastText vectors][5]. The first one converts them to a numpy array.
The second one exctractes a subset of most commonly used words to limit the size
of the resulting array.

Embeddings Tester
-----------------

`test-embedding.py` is an utility that loads the embedding and finds the closest
vectors (by cosine distance) to the given one. It's able to handle linear
combinations of vectors.

    ]==> ./test-embedding.py wiki.en.npy
    [i] Loading embeddings from: wiki.en.npy
    [i] Loading metadata from:   wiki.en.pkl
    [i] Number of tokens:        2519370
    [i] Vector size:             300
    [i] Prompt (CTRL-D to quit): poland
    [i] Looking for words nearest to poland...
    [i]    poland (1.00000)
    [i]    polands (0.80848)
    [i]    poland/lithuania (0.78729)
    [i]    poland, (0.78122)
    [i]    }poland (0.76378)
    [i]    poland/warsaw (0.75755)
    [i]    władysławów (0.74764)
    [i] Prompt (CTRL-D to quit): berlin + poland - germany
    [i] Looking for words nearest to berlin + poland - germany...
    [i]    warsaw (0.79216)
    [i]    poland (0.74023)
    [i]    kraków (0.73900)
    [i]    przemyśl–warsaw (0.73456)
    [i]    poznań (0.73201)
    [i]    „kraków (0.72293)
    [i]    warsaw, (0.72162)

[1]:http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
[2]: https://arxiv.org/pdf/1301.3781.pdf
[3]: http://mattmahoney.net/dc/textdata.html
[4]: http://mattmahoney.net/dc/text8.zip
[5]: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
