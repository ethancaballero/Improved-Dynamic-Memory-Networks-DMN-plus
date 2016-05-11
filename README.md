# Implementation of DMN+ (Improved Dynamic Memory Networks by MetaMind) in Theano
This is an implementation of DMN+ (Improved Dynamic Memory Networks) from the paper by Xiong et al. at MetaMind, Dynamic Memory Networks for Visual and Textual Question Answering, [arXiv:1603.01417](http://arxiv.org/abs/1603.01417).

It's a fork of YerevaNN's implementation of the initial version of Dynamic Memory Networks.

Question answering webapp using implementation is currently running at [ethancaballero.pythonanywhere.com](http://ethancaballero.pythonanywhere.com/); type the number 2, 3, 6, or 17 into Task Type box and then click 'Load Task Type' button to start webapp.

## Repository contents

| file | description |
| --- | --- |
| `webapp.py` | run webapp demo of DMN+ (adapted from [MemN2N webapp](https://github.com/vinhkhuc/MemN2N-babi-python)) |
| `dmn_tied.py` | weights of answer module are tied; trains faster |
| `dmn_untied.py` | weights of answer module are untied; slightly better performance for most taks, but slower training |
| `utils.py` | tools for working with bAbI tasks |
| `nn_utils.py` | helper functions on top of Theano and Lasagne |
| `fetch_babi_data.sh` | shell script to fetch bAbI tasks (adapted from [MemN2N](https://github.com/npow/MemN2N)) |

## Usage

This implementation is based on Theano, Lasagne, and Keras. One way to install them is:

    pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
    pip install https://github.com/Lasagne/Lasagne/archive/master.zip
    pip install keras
    
numpy, scipy, & flask are installed with pip install *
    
Also, NLTK is used for sentence splitting. to install:

    pip install nltk
    
then Run the Python interpreter and type these commands to download punkt dataset:

    import nltk
    nltk.download('punkt')

The following bash scripts will download bAbI tasks.

    ./fetch_babi_data.sh

Use `main.py` to train a network:

    python main.py --network dmn_tied --mode train --babi_id 1

The states of the network will be saved in `states/` folder. 
There are pretrained states for bAbI tasks 2, 3, 6, & 17 in the states folder.

`.theanorc` file contains theano configuration that was used; dmn_tied might yield NaN if included `.theanorc` configuration is not used.
definitely make sure floatX is set equal to float32 in `.theanorc` file

## Webapp Usage
The webapp uses this implementation to answer questions about stories from bAbi qa task 2, 3, 6, or 17; these tasks were chosen because they are the tasks that receive the largest performance increase from DMN+ (and conversely are the tasks that initial DMN struggled with the most). The webapp does not offer server side training because I couldn't find a free hosting service that has sufficient compute to train in a reasonable amount of time.

The question answering webapp using this implementation is currently running at [ethancaballero.pythonanywhere.com](http://ethancaballero.pythonanywhere.com/). Upon arriving at the webapp page, type the number 2, 3, 6, or 17 into Task Type box and then click 'Load Task Type' button to load bAbi task of that type (loading the task usually takes about one minute). When loading finishes, the Story and Question text boxes will respectively be filled with a story and a question. Next, click 'Predict Answer' to view the network's answer_prediction, attentions, and confidence given the current story_question pair or click 'Load New Story' to get new story_question pair.  To load bAbi tasks of a different type, type the number 2, 3, 6, or 17 into Task Type box and then click 'Load Task Type' button to load bAbi task of that type.

To run your own webapp locally:

    python webapp.py

then go to http://0.0.0.0:5000/ in browser 

##Additions implemented in this DMN+ repo:
*positional sentence encoder produced by: fi = sum(lj * wij) in M through j=1
where lj is a column vector with structure ljd = (1 - j/M) - (d/D)(1 - 2j/M)
*Bidirectional GRU for input fusion layer of input module
*gt from Episodic Memory Module is a function similar to softmax
*Attention based GRU in which update gate u of GRU is replaced with gt to yield this hidden layer: hi=git  h?i +(1 git) hi 1
*untied answer module weights (this addition was not implemented before training time so it is not used by the webapp)

*most of the additions are in lines ~250-440 of dmn_*.py

## Questions
* Unsure how/where input dropout is supposed to be implemented
* How is the ReLU memory update layer supposed to work?? The paper seems to use concatenated floats to allocate the subtensors of Wt, but how can float(s) allocate subtensor(s) (wouldn't integers need to be used?).

## Benchmarks
* DMN+ implementation accuracy on test data is higher than that of 2015 DMN, but lower than accuracies reported in DMN+ paper. I'm pretty sure this insufficient generalization is due to an error in my implementation of dropout on the initial sentence encodings.

| task (10k) | Test error rates |
| --- | --- |
| 2: 2 supporting facts | 9.5 |
| 3: 3 supporting facts | 29.6 |
| 6: yes/no questions | 0.3 |
| 17: positional reasoning | 20.2 |

training settings were: adam with lr=.0002 and beta=.5; dropout p=.1; l2 reg = .0005; ~100 epochs

## TODO
* Mini-batch training
* Implement Visual Portion 
* figure out update ReLU and input dropout

## Acknowledgements
* This is a fork of [YerevaNN's implementation of the initial version of Dynamic Memory Networks](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano).
* Webapp design is based off [Vinh Khuc's MemN2N Webapp implementation](https://github.com/vinhkhuc/MemN2N-babi-python)

## License?
JSON
