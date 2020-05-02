# Welcome to the NUBIA repo! 

## NUBIA (NeUral Based Interchangeability Assessor) is a new SoTA evaluation metric for text generation

Check out the [paper](https://arxiv.org/abs/2004.14667), [blog post](https://wl-research.github.io/blog/), [FAQ](https://github.com/wl-research/nubia/blob/master/FAQ.md) and [demo colab notebook](https://colab.research.google.com/drive/1_K8pOB8fRRnkBPwlcmvUNHgCr4ur8rFg).

<center><img src="images/contradiction-demo.gif"></img></center>

#### Installation:

Clone the repo: `git clone https://github.com/wl-research/nubia.git`

Install requirements: `pip install -r requirements.txt`

#### Use:

Import and initialize the `Nubia` class from `nubia.py` (wherever you cloned the repo):

Note: The first time you initialize the class it will download the pretrained models from the S3 bucket, this could take a while depending on your internet connection.

`Nubia().score` takes seven parameters: `(ref, hyp, verbose=False, get_features=False, six_dim=False, aggregator="agg_two")`

`ref` and `hyp` are the strings nubia will compare. 

Setting `get_features` to `True` will return a dictionary with additional features (semantic relation, contradiction, irrelevancy, logical agreement, and grammaticality) aside from the nubia score. `Verbose=True` prints all the features.

`six_dim = True` will not take the word count of `hyp` and `ref` into account when computing the score.

`aggregator` is set to `agg_two` by default, but you may choose to try `agg_one` which was used to achieve the WMT 2017 results.

#### Example:

`nubia.score("The dinner was delicious.", "The dinner did not taste good.", verbose=True, get_features=True)`
```
Semantic relation: 1.4594818353652954/5.0
Percent chance of contradiction: 99.90345239639282%
Percent chance of irrelevancy or new information: 0.06429857457987964%
Percent chance of logical agreement: 0.03225349937565625%


NUBIA score: 0.18573102718477918/1.0
```
See more exmaples of usage at our [demo notebook](https://github.com/wl-research/nubia/blob/master/nubia-demo.ipynb) `nubia-demo.pynb`

#### Citation:

If you use Nubia in your work, please cite: 

```
@misc{kane2020nubia,
    title={NUBIA: NeUral Based Interchangeability Assessor for Text Generation},
    author={Hassan Kane and Muhammed Yusuf Kocyigit and Ali Abdalla and Pelkins Ajanoh and Mohamed Coulibali},
    year={2020},
    eprint={2004.14667},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

#### Contact Us: 

You can reach us by email [here](mailto:hassanmohamed@alum.mit.edu) or by opening an issue at this repo. 
