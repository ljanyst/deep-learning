
A word-wise RNN
===============

Train an word-wise RNN on a text body and generate a similar text based on the
knowledge. I tried training it on the whole [dataset][1] of dialogs from The
Simpsons but was not able to make the learning process converge with anything
that would compute in a reasonable time. That's something to be investigated
further. Here's what I got from the subset of this data (Moe's Tavern):

    ]==> ./generate.py --name simpsons-moe --samples 500 --prime bart_simpson
    Project name:         simpsons-moe
    Network checkpoint:   simpsons-moe/final.ckpt
    LSTM size:            2048
    LSTM layers:          3
    Embedding size:       200
    Priming text:         bart_simpson
    Samples to generate:  200
    [i] Restoring a checkpoint from simpsons-moe/final.ckpt
    bart_simpson... (really, by sigh) " homer_simpson, i'm going to be a place dive.
    moe_szyslak: i got it used from the navy. you can flash-fry a buffalo in forty seconds.
    homer_simpson: forty seconds?   (whining) but i want it.

    homer_simpson: (ringing bell) hear ye, hear ye, my daughter has something to tell you about jebediah springfield.
    moe_szyslak: aw, the evening bedtime readin'.
    moe_szyslak: (snorts) nobody does.
    kemi: (portuguese) eu não quero dizer para mostrar (french) je ne veux pas montrer (spanish) no, sad, not this again.
    moe_szyslak: what?   it's 'cause of her i put in a bidet. well, it's actually just a step ladder by the water fountain.
    homer_simpson: now, you learn your numbers from perfect.
    bart_simpson: oh, yeah, can i look too?
    moe_szyslak: sure, but it'll cost somethin' how to make a job?

[1]: https://kaggle.com/wcukierski/the-simpsons-by-the-data
