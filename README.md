# Welcome to the NUBIA repo! 

## NUBIA (NeUral Based Interchangeability Assessor) is a SoTA text similarity metric.

Check out the [paper](https://arxiv.org/abs/2004.14667) on ArXiv, and this [blog post](https://wl-research.github.io/blog/2020/04/29/introducing-nubia.html).

#### Installation:

Clone the repo: `git clone git@github.com:wl-research/nubia.git`

Installing requirements: `pip install -r requirements.txt`

#### Use:

Import and initialize the `Nubia` class from `nubia.py` (wherever you cloned the repo):

Note: The first time you initialize the class it will download the pretrained models from the S3 bucket, this could take a while depending on your internet connection.

`Nubia().score` takes seven parameters: `(ref, hyp, verbose=False, get_features=False, six_dim=False, aggregator="agg_one")`

`ref` and `hyp` are the strings nubia will compare. 

Setting `get_features` to `True` will return a dictionary with additional features (semantic relation, contradiction, irrelevancy, logical agreement, and grammaticality) aside from the nubia score. `Verbose=True` prints all the features.

`six_dim = True` will not take the word count of `hyp` and `ref` into account when computing the score.

`aggregator` is set to `agg_one` by default, but you may choose to try `agg_two` which is

#### Example:

`nubia.score("The dinner was delicious.", "The dinner did not taste good.", verbose=True, get_features=True)`

Semantic relation: 1.4594818353652954/5.0

Percent chance of contradiction: 99.90345239639282%

Percent chance of irrelevancy 0.06429857457987964%

Percent chance of logical agreement 0.03225349937565625%

Grammaticality score for reference sentence: 5.1724853515625

Grammaticality score for candidate sentence:  4.905452728271484

NUBIA score: 0.18573102718477918/1.0

See more exmaples of usage at our [demo notebook](https://github.com/wl-research/nubia/blob/master/nubia-demo.ipynb) `nubia-demo.pynb`

#### Citation:

If you use Nubia in your work, please cite it as: 

arXiv:2004.14667
