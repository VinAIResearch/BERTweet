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

## <a name="exp"></a> Main results 

<img width="257" alt="postagging" src="https://user-images.githubusercontent.com/2412555/82403966-490d6380-9a8a-11ea-8530-18d6e06641cf.png"><img width="250" alt="ner" src="https://user-images.githubusercontent.com/2412555/82403965-4874cd00-9a8a-11ea-9847-0192d11e3e31.png"><img width="250" alt="textclassification" src="https://user-images.githubusercontent.com/2412555/82403956-43b01900-9a8a-11ea-8b34-a89e1e7d52a7.png">

## <a name="fairseq"></a> Using BERTweet in [`fairseq`](https://github.com/pytorch/fairseq) 

### <a name="install1"></a> Installation 

 -  Python version >= 3.6
 - [PyTorch](http://pytorch.org/) version >= 1.4.0
 - [`fairseq`](https://github.com/pytorch/fairseq)
 - `fastBPE`: `pip3 install fastBPE`

### <a name="models1"></a> Pre-trained model 

Model | #params | size | Download
---|---|---|---
`BERTweet-base` | 135M | 1.2GB | [BERTweet_base_fairseq.tar.gz](https://public.vinai.io/BERTweet_base_fairseq.tar.gz) (`md5sum` 692cd647e630c9f5de5d3a6ccfea6eb2)

 - `wget https://public.vinai.io/BERTweet_base_fairseq.tar.gz`
 - `tar -xzvf BERTweet_base_fairseq.tar.gz`

### <a name="usage1"></a> Example usage 

```python
import torch

# Load BERTweet-base in fairseq
from fairseq.models.roberta import RobertaModel
BERTweet = RobertaModel.from_pretrained('/Absolute-path-to/BERTweet_base_fairseq', checkpoint_file='model.pt')
BERTweet.eval()  # disable dropout (or leave in train mode to finetune)

# Incorporate the BPE encoder into BERTweet-base 
from fairseq.data.encoders.fastbpe import fastBPE  
from fairseq import options  
parser = options.get_preprocessing_parser()  
parser.add_argument('--bpe-codes', type=str, help='path to fastBPE BPE', default="/Absolute-path-to/BERTweet_base_fairseq/bpe.codes")  
args = parser.parse_args()  
BERTweet.bpe = fastBPE(args) #Incorporate the BPE encoder into BERTweet

# INPUT TEXT IS TOKENIZED!
line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"  

# Extract the last layer's features  
subwords = BERTweet.encode(line)  
last_layer_features = BERTweet.extract_features(subwords)  
assert last_layer_features.size() == torch.Size([1, 21, 768])  
  
# Extract all layer's features (layer 0 is the embedding layer)  
all_layers = BERTweet.extract_features(subwords, return_all_hiddens=True)  
assert len(all_layers) == 13  
assert torch.all(all_layers[-1] == last_layer_features)  

# Filling marks  
masked_line = 'SC has first two presumptive cases of  <mask> , DHEC confirms HTTPURL via @USER :cry:'  
topk_filled_outputs = BERTweet.fill_mask(masked_line, topk=5)  
for candidate in topk_filled_outputs:  
    print(candidate)
    # ('SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:', 0.8643638491630554, 'coronavirus')
    # ('SC has first two presumptive cases of Coronavirus , DHEC confirms HTTPURL via @USER :cry:', 0.04520644247531891, 'Coronavirus')
    # ('SC has first two presumptive cases of #coronavirus , DHEC confirms HTTPURL via @USER :cry:', 0.035870883613824844, '#coronavirus')
    # ('SC has first two presumptive cases of #COVID19 , DHEC confirms HTTPURL via @USER :cry:', 0.029708299785852432, '#COVID19')
    # ('SC has first two presumptive cases of #Coronavirus , DHEC confirms HTTPURL via @USER :cry:', 0.005226477049291134, '#Coronavirus')


```

### <a name="preprocess"></a> A script to pre-process raw input Tweets 

Before applying `fastBPE` to the pre-training corpus of 850M English Tweets, we tokenized these  Tweets using `TweetTokenizer` from the NLTK toolkit and used the `emoji` package to translate emotion icons into text strings (here, each icon is referred to as a word token).   We also normalized the Tweets by converting user mentions and web/url links into special tokens `@USER` and `HTTPURL`, respectively. Thus it is recommended to also apply the same pre-processing step for BERTweet-based downstream applications w.r.t. the raw input Tweets.

- Installation: `pip3 install nltk emoji`

```python
from TweetNormalizer import normalizeTweet
print(normalizeTweet("SC has first two presumptive cases of coronavirus, DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier"))
```

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


