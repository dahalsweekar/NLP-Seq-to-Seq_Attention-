# NLP-Seq-to-Seq_Attention

## Introduction
This repository offers an implementation of text recognition where lstm network is used for seq-to-seq text recognition. Resnet-18 backbone is used for feature extraction.
Project is followed based upon PyTorch tutorial.

### Encoder Network
The encoder of a seq2seq network is a RNN that outputs some value for every word from the input sentence. For every input word the encoder outputs a vector and a hidden state, and uses the hidden state for the next input word.

![encoder-network](https://github.com/dahalsweekar/NLP-Seq-to-Seq_Attention-/assets/99968233/3b9b52f8-006c-48d2-aa73-5c1270da2fd9)
### Decoder Network
In the simplest seq2seq decoder we use only last output of the encoder. This last output is sometimes called the context vector as it encodes context from the entire sequence. This context vector is used as the initial hidden state of the decoder.

![decoder-network](https://github.com/dahalsweekar/NLP-Seq-to-Seq_Attention-/assets/99968233/bac5d1a7-c5b6-4d86-88b8-761827f8c4ed)
### Attention Network
Attention allows the decoder network to “focus” on a different part of the encoder’s outputs for every step of the decoder’s own outputs. First we calculate a set of attention weights. These will be multiplied by the encoder output vectors to create a weighted combination. The result (called attn_applied in the code) should contain information about that specific part of the input sequence, and thus help the decoder choose the right output words.

![attention-decoder-network](https://github.com/dahalsweekar/NLP-Seq-to-Seq_Attention-/assets/99968233/4a20b502-c97a-475d-b60d-b025b14de738)

## Features

  ### Training/Evaluation

| Flags  | Usage |
| ------------- | ------------- |
| ```--eval``` | run eval | 
| ```--lang1```  | language 1	|                                                                   
| ```--lang2```  | language 2 |
| ```--reverse```  | reverse  | 
| ```--max_length```  | length of filter | 
| ```--epoch```  | Set number of epochs  |
| ```--batch_size```  | set batch size  |
| ```--data_path```  | data to .txt file  |

## Installation
  ### Requirements
    -Python3
    -Cuda

  ### Install
    1. git clone https://github.com/dahalsweekar/NLP-Seq-to-Seq_Attention-.git

## Dataset sample:
```
['Tom only has one shoe on.', 'टमले एउटा मात्र जुत्ता लगाएको छ।']
```
## Training 
 ```
 python main.py --epoch <set>
 ```

## Evaluation

 ```
 python main.py --eval
 ```

