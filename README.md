  
#### Table of contents
1. [Introduction](#introduction)
2. [Using BERTweet with `transformers`](#transformers)
    - [Installation](#install2)
    - [Pre-trained model](#models2)
    - [Example usage](#usage2)
    - [Normalize raw input Tweets](#preprocess)
3. [Using BERTweet with `fairseq`](#fairseq)


# <a name="introduction"></a> BERTweet: A pre-trained language model for English Tweets 

 - BERTweet is the first public large-scale language model pre-trained for English Tweets. BERTweet is trained based on the [RoBERTa](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md)  pre-training procedure, using the same model configuration as [BERT-base](https://github.com/google-research/bert). 
 - The corpus used to pre-train BERTweet consists of 850M English Tweets (16B word tokens ~ 80GB), containing 845M Tweets streamed from 01/2012 to 08/2019 and 5M Tweets related the **COVID-19** pandemic. 
 - BERTweet does better than its competitors RoBERTa-base and [XLM-R-base](https://arxiv.org/abs/1911.02116) and outperforms previous state-of-the-art models on three downstream Tweet NLP tasks of Part-of-speech tagging, Named entity recognition and text classification.

The general architecture and experimental results of BERTweet can be found in our [paper](https://arxiv.org/abs/2005.10200):

    @inproceedings{bertweet,
    title     = {{BERTweet: A pre-trained language model for English Tweets}},
    author    = {Dat Quoc Nguyen and Thanh Vu and Anh Tuan Nguyen},
    booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
    year      = {2020}
    }

**Please CITE** our paper when BERTweet is used to help produce published results or is incorporated into other software.

## <a name="transformers"></a> Using BERTweet with `transformers`

### <a name="install2"></a> Installation 

 -  Python version >= 3.6
 - [PyTorch](http://pytorch.org/) version >= 1.4.0
 -  Install `transformers` from our development branch:
    - `git clone https://github.com/datquocnguyen/transformers.git`
    - `cd transformers`
    - `pip3 install .`
 - Install `emoji`: `pip3 install emoji`

We also created a pull request to integrate BERTweet into the master branch of the `transformers` library. Please see the latest updates at:  https://github.com/huggingface/transformers/pull/6129

### <a name="models2"></a> Pre-trained model 


Model | #params | Arch. | Pre-training data
---|---|---|---
`vinai/bertweet-base` | 135M | base | 845M English Tweets (80GB)


### <a name="usage2"></a> Example usage 


```python
import torch
from transformers import AutoModel, AutoTokenizer #, BertweetTokenizer

bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
#tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base")

# INPUT TWEET IS ALREADY NORMALIZED!
line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"

input_ids = torch.tensor([tokenizer.encode(line)])

with torch.no_grad():
    features = bertweet(input_ids)  # Models outputs are now tuples
```

### <a name="preprocess"></a> Normalize raw input Tweets 

Before applying `fastBPE` to the pre-training corpus of 850M English Tweets, we tokenized these  Tweets using `TweetTokenizer` from the NLTK toolkit and used the `emoji` package to translate emotion icons into text strings (here, each icon is referred to as a word token).   We also normalized the Tweets by converting user mentions and web/url links into special tokens `@USER` and `HTTPURL`, respectively. Thus it is recommended to also apply the same pre-processing step for BERTweet-based downstream applications w.r.t. the raw input Tweets.

```python
import torch
from transformers import BertweetTokenizer

# Load the BertweetTokenizer with a normalization mode if the input Tweet is raw
tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

# BERTweet's tokenizer can be also loaded in the "Auto" mode
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

line = "SC has first two presumptive cases of coronavirus, DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier"

input_ids = torch.tensor([tokenizer.encode(line)])
```

## <a name="fairseq"></a> Using BERTweet with `fairseq`

Please see details at [HERE](https://github.com/VinAIResearch/BERTweet/blob/master/README_fairseq_and_old_transformers_version.md)!

## License
    
    MIT License

    Copyright (c) 2020 VinAI Research

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.


