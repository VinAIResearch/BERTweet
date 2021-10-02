  
#### Table of contents
1. [Introduction](#introduction)
2. [Main results](#results)
3. [Using BERTweet with `transformers`](#transformers)
    - [Pre-trained models](#models2)
    - [Example usage](#usage2)
    - [Normalize raw input Tweets](#preprocess)
4. [Using BERTweet with `fairseq`](#fairseq)


# <a name="introduction"></a> BERTweet: A pre-trained language model for English Tweets 

BERTweet is the first public large-scale language model pre-trained for English Tweets. BERTweet is trained based on the [RoBERTa](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md)  pre-training procedure. The corpus used to pre-train BERTweet consists of 850M English Tweets (16B word tokens ~ 80GB), containing 845M Tweets streamed from 01/2012 to 08/2019 and 5M Tweets related to the **COVID-19** pandemic. The general architecture and experimental results of BERTweet can be found in our [paper](https://aclanthology.org/2020.emnlp-demos.2/):

    @inproceedings{bertweet,
    title     = {{BERTweet: A pre-trained language model for English Tweets}},
    author    = {Dat Quoc Nguyen and Thanh Vu and Anh Tuan Nguyen},
    booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
    pages     = {9--14},
    year      = {2020}
    }

**Please CITE** our paper when BERTweet is used to help produce published results or is incorporated into other software.

## <a name="results"></a> Main results

<img width="275" alt="postagging" src="https://user-images.githubusercontent.com/2412555/135724590-01d8d435-262d-44fe-a383-cd39324fe190.png"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img width="275" alt="ner" src="https://user-images.githubusercontent.com/2412555/135724598-1e3605e7-d8ce-4c5e-be4a-62ae8501fae7.png">   

<img width="275" alt="sentiment" src="https://user-images.githubusercontent.com/2412555/135724597-f1981f1e-fe73-4c03-b1ff-0cae0cc5f948.png"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img width="275" alt="irony" src="https://user-images.githubusercontent.com/2412555/135724595-15f4f2c8-bbb6-4ee6-82a0-034769dec183.png">


## <a name="transformers"></a> Using BERTweet with [`transformers`](https://github.com/huggingface/transformers)

### <a name="models2"></a> Pre-trained models 



Model | #params | Arch. | Pre-training data
---|---|---|---
`vinai/bertweet-base` | 135M | base | 850M English Tweets (cased)
`vinai/bertweet-covid19-base-cased` | 135M | base | 23M COVID-19 English Tweets (cased)
`vinai/bertweet-covid19-base-uncased` | 135M | base | 23M COVID-19 English Tweets (uncased)
`vinai/bertweet-large` | 355M | large | 873M English Tweets (cased) 

- 09/2020: Two pre-trained models `vinai/bertweet-covid19-base-cased` and `vinai/bertweet-covid19-base-uncased` are resulted by further pre-training the pre-trained model `vinai/bertweet-base` on a  corpus of 23M COVID-19 English Tweets.
- 08/2021: Released `vinai/bertweet-large`.

### <a name="usage2"></a> Example usage 


```python
import torch
from transformers import AutoModel, AutoTokenizer 

bertweet = AutoModel.from_pretrained("vinai/bertweet-large")

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large")

# INPUT TWEET IS ALREADY NORMALIZED!
line = "DHEC confirms HTTPURL via @USER :crying_face:"

input_ids = torch.tensor([tokenizer.encode(line)])

with torch.no_grad():
    features = bertweet(input_ids)  # Models outputs are now tuples
    
## With TensorFlow 2.0+:
# from transformers import TFAutoModel
# bertweet = TFAutoModel.from_pretrained("vinai/bertweet-large")
```

### <a name="preprocess"></a> Normalize raw input Tweets 

Before applying BPE to the pre-training corpus of English Tweets, we tokenized these  Tweets using `TweetTokenizer` from the NLTK toolkit and used the `emoji` package to translate emotion icons into text strings (here, each icon is referred to as a word token).   We also normalized the Tweets by converting user mentions and web/url links into special tokens `@USER` and `HTTPURL`, respectively. Thus it is recommended to also apply the same pre-processing step for BERTweet-based downstream applications w.r.t. the raw input Tweets. 

#### Internal normalizer

BERTweet provides this pre-processing step by enabling the `normalization` argument of its tokenizer. Note that this argument currently only supports models "`vinai/bertweet-base`", "`vinai/bertweet-covid19-base-cased`" and "`vinai/bertweet-covid19-base-uncased`".

 - Install `emoji`: `pip3 install emoji`

```python
import torch
from transformers import AutoTokenizer

# Load the AutoTokenizer with a normalization mode if the input Tweet is raw
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

# from transformers import BertweetTokenizer
# tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

line = "DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier 😢"

input_ids = torch.tensor([tokenizer.encode(line)])
```

#### External normalizer

Alternatively, given the raw input Tweets, to obtain the same pre-processing output, users could employ our  [TweetNormalizer](https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py) module.

- Installation: `pip3 install nltk emoji`

```python
import torch
from transformers import AutoTokenizer
from TweetNormalizer import normalizeTweet

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large")

line = normalizeTweet("DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier 😢")

input_ids = torch.tensor([tokenizer.encode(line)])
```

## <a name="fairseq"></a> Using BERTweet with `fairseq`

Please see details at [HERE](https://github.com/VinAIResearch/BERTweet/blob/master/README_fairseq.md)!

## License
    
    MIT License

    Copyright (c) 2020-2021 VinAI Research

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


