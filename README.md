# NLP-Seq-to-Seq_Attention

## Introduction
This repository offers an implementation of text recognition where lstm network is used for seq-to-seq text recognition. Resnet-18 backbone is used for feature extraction.

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

