
Sorting with an Encoder-Decoder RNN
===================================

Generate 10000000 o random strings of characters of length from 1 to 10 and
their sorted counterparts. Then, use the strings to train an encoder-decoder
recurrent neural network so that for a given input sentence it produces its
sorted counterpart.

    ]==> ./sort.py --seq=hello   
    [i] Project name:         test
    [i] Network checkpoint:   test/final.ckpt
    [i] LSTM size:            64
    [i] LSTM layers:          2
    [i] Max sequence length:  10
    [i] Sequence to sort:     hello
    [i] Restoring a checkpoint from test/final.ckpt
    [i] Sorted sequence:      ehllo

    ]==> ./sort.py --seq=please
    [i] Project name:         test
    [i] Network checkpoint:   test/final.ckpt
    [i] LSTM size:            64
    [i] LSTM layers:          2
    [i] Max sequence length:  10
    [i] Sequence to sort:     please
    [i] Restoring a checkpoint from test/final.ckpt
    [i] Sorted sequence:      aeelps
