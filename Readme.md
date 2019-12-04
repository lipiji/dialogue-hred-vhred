# HRED VHRED VHCR
modified based on： https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling



## Preprocess data
./data: train.txt, dev.txt, test.txt

format: u1 </s> u2 </s> \t response

example: w11 w12 w13 </s> w21 w22 </s> w31 w32 w33 w34 \t w1 w2 w3



## Training
Go to the model directory and set the save_dir in configs.py (this is where the model checkpoints will be saved)

We provide our implementation of VHCR, as well as our reference implementations for [HRED](https://arxiv.org/abs/1507.02221) and [VHRED](https://arxiv.org/abs/1605.06069).

To run training:
```
python train.py --model=<model> --batch_size=<batch_size>
```

For example:
1. Train HRED:
```
python train.py  --model=HRED
```

2. Train VHRED with word drop of ratio 0.25 and kl annealing iterations 250000:
```
python train.py --model=VHRED --batch_size=40 --word_drop=0.25 --kl_annealing_iter=250000
```

3. Train VHCR with utterance drop of ratio 0.25:
```
python train.py --model=VHCR --batch_size=40 --sentence_drop=0.25 --kl_annealing_iter=250000
```




## Evaluation
To evaluate the word perplexity:
```
python eval.py --model=<model> --checkpoint=<path_to_your_checkpoint>
```

For embedding based metrics, you need to download [Google News word vectors](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing), unzip it and put it under the datasets folder.
Then run:
```
python eval_embed.py --model=<model> --checkpoint=<path_to_your_checkpoint>
```


## Reference

If you use this code or dataset as part of any published research, please refer the following paper.

```
@inproceedings{VHCR:2018:NAACL,
    author    = {Yookoon Park and Jaemin Cho and Gunhee Kim},
    title     = "{A Hierarchical Latent Structure for Variational Conversation Modeling}",
    booktitle = {NAACL},
    year      = 2018
    }
```
