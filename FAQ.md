# Frequently Asked Questions

### What is the TLDR for what NUBIA is?

For a given reference sentence, NUBIA takes as input a candidate sentence and scores it on a 0-1 scale on how good of a substitute/replacement the candidate sentence is for the reference sentence.
Think of it as an exam. If NUBIA has just one answer to the exam question (assuming it’s not too complex for now), it can grade candidate sentences better than currently published metrics. 
A long-term advantage NUBIA has is that all of its core components are learnt. It is therefore expected to continuously improve over time in synergy with NLP research and human evaluation efforts.

### What is the difference between what NUBIA measures and just regular semantic similarity?
Roughly speaking, semantic similarity measures a score along the lines of “over the space of all possible sentences, how relatively similar are these two?”
On the other end, Nubia measures a score along the lines of “is the candidate sentence a good substitute for that specific reference sentence?” Practically, it makes NUBIA way more sensitive to small variations in nuances/meaning and grammatical mistakes compared with typical semantic similarity scores. It actually uses semantic similarity as a neural feature along with other ones.

### I see NUBIA is entirely made of deep learning models and neural networks. Is it a giant black box?

NUBIA scores are interpretable. If you check the examples provided,  the neural features provide sub-score on whether the sentence pairs are semantically related, in logical contradiction, agreement or unrelated. The grammaticality features are a bit harder to interpret but one can get an intuition by providing gibberish sentences and see how they change.

### Why is there a need to design new evaluation metrics for text generation tasks?

Unlike machine learning tasks such as classification and regression, text generation (i.e. machine translation, summarization, image captioning) is very nuanced and the gold standard to judge models is well conducted human evaluation. However, we can’t afford that for most models so instead, proxies in the form of machine learning models were designed to approximate human judgment of quality. A consequence of this unique setup is that until we get quasi-human judgment of quality (which we’re still far from) the metrics themselves have to be upgraded every couple years to reflect the dynamic progress of the field. 

### What’s the difference between WMT17 and WMT18-19 setup?

This is explained in the paper but in short, the WMT17 dataset and previous years focused on direct assessment of sentence quality on a scale of 0-100 where each sentence was scored by 15 human evaluators. The metrics were scored on how well they directly correlated with that average human judgement score. To clarify, we also use the latest available metrics (as of May 2020) and their performance on that benchmark so while the dataset is from 2017, all the tested metrics are the best available.
WMT18 and WMT19 benchmarks focused on relative ranking of different sentences (given a reference sentence, and multiple candidate texts, pick which candidate text is better). The low scores achieved by all metrics on this task reveal that their relative perception of quality differs from ours (the metrics’ favorite candidate sentences may not be the one favored by human evaluators). However, much higher correlation with absolute scores mean that they very well agree with human evaluators on identifying good and bad substitutions.

### As an ML/NLP Researcher, how do I use NUBIA in my work?

NUBIA can be used to score the quality of image caption/translation/summarization after training your text generation models.
We suspect NUBIA can be used as a component of a loss function to give feedback to decoder architecture during training assuming that reference sentences are available during training.

### What research directions have the greatest potential to further improve NUBIA?

There are many!
- Incorporating or building (neural) feature extractors capturing grammaticality and sentence acceptability (detecting gibberish) is currently area where a lot can still be done (we’re looking into this)
- Extending performance studies beyond image captioning, machine translation and extending performance measures beyond correlation with human judgement (we’re also actively working on that)
- Using multilingual models to go beyond scoring english sentences,
- Adopting or implementing sound standards for naming/versioning and model reporting. NUBIA is made of 3 deep learning models and one neural network. Giving it a “serial number” could help for reproducibility. Hardware or Electrical Engineering background would greatly help with this.
- Systematically studying the emergent behavior of NUBIA and characterizing how it reacts to changes in modification of part of speech (personal pronoun, verbs, addition of adverbs etc) that subtly change sentence meanings is an area where we have barely scratched the surface. Having a linguistics background would be great to study this. The notebooks also enable anyone, even with a nontechnical background, to dive into this.
- One NUBIA call requires computation from 3 Transformers and 1 neural network. For interesting reference and candidate examples, visualizing the computations done at each neural feature extractor and then how they are fed to the aggregator can allow to investigate the full chain of reasoning leading to these scores. Probing the attention weights of each neural feature extractor is a good start but the interesting thing here is that the aggregator “couples” these different transformers which is why a single change of word or the addition of a “not” resets the entire computation.
- The current, full NUBIA model is 5 GB and a bit slow. Working on model pruning/quantization will have a great impact on downstream adoption of NUBIA as an evaluation metric. There is already a good amount of work in the space.
- Transfer learning setups are also interesting. For the image captioning experiments, we used the same model trained on machine translation quality and showed it outperforms metrics currently used and specifically designed for image captioning on agreeing with human judgement. What does adapting NUBIA to score Chest X Ray radiology reports (given gold reference ones) where grammaticality doesn’t matter as much look like? (Hint: we’re currently investigating this). 
- We are not sure whether there is an algorithm by which a computational or human adversary can systematically fool NUBIA. More specifically: for any given sentence X, a threshold epsilon, is there a procedure by which a reference sentence Y can be a meaningless substitute but still fool NUBIA into giving it an extremely high score  (above 1 - epsilon)? Empirically, we have played around with the model and are confident that , for epsilon values around 0.1 to 0.3 , a human adversary can do a decent job at introducing input perturbations that make the candidate sentence look gibberish or at least have typos and still achieve a NUBIA score above 1 - epsilon. We are not sure if this can happen for arbitrarily large epsilon values. We are also not sure what a systematic computational procedure would look like in a black-box and white-box setting. ML security researchers are welcomed to more rigorously formalize the threat models and problem setting based on the description above.

### How can the results be reproduced?

The models and experimental setups are extensively described in the paper. The notebooks and code are published. The datasets we have used (WMT & Flickr) are all publicly available and mentioned in the paper. That said, to improve convenience and streamline reproducibility, we will also add them on the repo with a colab notebook to replicate results (ETA ± May 15th 2020). 

### I would like to get directly involved. What do you suggest?

A good background in probability, multivariable calculus, programming, machine learning and NLP are essential. If you want to pick an area of your own, some of the research questions highlighted above can be a good start. If you want to directly collaborate with us, reach out [here](mailto:hassanmohamed@alum.mit.edu) with your background and motivation! To set expectations, you must be willing to at least commit 12-15h a week, for a period of at least 3 months to engage in direct collaboration. That said, we have way more research questions than we can handle at this point so extra help wouldn’t hurt :)
