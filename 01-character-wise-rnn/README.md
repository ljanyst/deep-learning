
A character-wise RNN
====================

The programs in this directory implement a character-wise recurrent neural
networks. The implemetation is based on the one presented [here][1]. One can
train a network on a body of text and then use it to generate text in similar
style.

That's what the network generated after seeing "Anna Karenina":

    ]==> ./generate.py --name=anna-karenina --checkpoint=-1 --prime="The " --samples=1000 --top-predictions=3
    Project name:                     anna-karenina
    Network checkpoint:               anna-karenina/final.ckpt
    LSTM size:                        1024
    LSTM layers:                      3
    Priming text:                     The
    Samples to generate:              1000
    Number of top preductions to use: 3
    [i] Restoring a checkpoint from anna-karenina/final.ckpt
    The steps, and the party
    started at the dining room, and the strange conversation was standing
    with a smile. The same words, the same though, and to bring a little
    shoulder, and the same smile, and the same strange face with which
    the priest and the past towards her, as though they were at the table,
    where a little girl would see how they had been the baby to them, and
    together at the table.

    "I'll tell you what I wanted to go to the country."

    "Well, then, I won't believe it," he said. "What am I to do? I want to
    do them, because I wanted to say..."

    "Yes," he said.

    "Well, then I'm natural and delicious, and I should not have thought of
    me, but that I could be so great the sake of me. If I do not ask you, to
    be a stranger and delightful tea on. The peasant was at a serious and
    the salary of him. It was an acquaintance."

    "Yes, yes," answered Levin, and stopped at the time with a
    smile, and the point was the only woman of the steps with his
    sharp arm and the carriage and the sharp health o

That's what it produced after seeing "Pan Tadeusz":

    ]==> ./generate.py --name=pan-tadeusz --checkpoint=-1 --prime="Pan" --samples=1000 --top-predictions=1
    Project name:                     pan-tadeusz
    Network checkpoint:               pan-tadeusz/final.ckpt
    LSTM size:                        1024
    LSTM layers:                      3
    Priming text:                     Pan
    Samples to generate:              1000
    Number of top preductions to use: 1
    [i] Restoring a checkpoint from pan-tadeusz/final.ckpt
    Pan Sędzia,
    Który stojąc na świecie powieści i zdrowie!
    Byłem to w tym domuciej i w szlachty i musię
    Podobna do Litwina stary i zabije,
    To poznać, że ona ciebie nie zaszko.
    Bo dziś zaczęła szlachtę; wszystko się zdawało,
    Że Wojski wciąż gra jeszcze, a to echo grało.
    Wyrawia szczęście, w której krzyknął: «Panowie,
    Bo gdyby zaczęłowski strzelców i zabijem,
    Bo wszyscy za nim stała, z powróci się z drzewa,
    Głowę od rani powieść, pomiędzy uczuciem,
    Aż wygrał się za połu, przez okna wyprawia,
    Czy wpadł do karczmy, znajdzie po pana i trudnie,
    A po chwili uciekał się z wolna i postawił,
    I sprawił się i pani i wszystkie sposobi,
    Upadają od ławę podniósł do sądiedzi,
    I choć się uszczupiona i czas jako wiecze,
    I posiedzi na zamku powiada z podarem
    Karczmy, tam się z podniesien do swego szlachcica:
    Tak jako wielko pance, zwykła gromadarza,
    Że przyszło uciekła w tym konej przy pokocy.
    Prosto u nim stał z wiatrem i głowę zadziwił.
    «Bracie — rzekł — odobiegło w nim się powiekami,
    Że nas wiesz, może było

[1]: https://github.com/udacity/deep-learning/tree/master/intro-to-rnns
