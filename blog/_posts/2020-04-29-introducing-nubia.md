# Introducing NUBIA

# Introducing NUBIA: 
**N**e**u**ral **B**ased **I**nterchangeability **A**ssessor

Here's the table of contents:

1. TOC
{:toc}


## NUBIA = Neural (UnBiased/Unified ; Basic/Better) Interchangeability Assessor

### TLDR: 

We have designed a new, fully neural paradigm to build evaluation metrics for language generation tasks. Our proof of concept is interpretable, modular, largely exceeds currently used metrics and slightly exceeds/matches state-of-the art as far as correlation with human judgment of quality goes while presenting upsides of continual improvement. This has the potential to remove bottlenecks and spur progress in the task of machine translation, image captioning and summarization.

![](https://wl-research.github.io/blog/images/birds-nubia.png)


### Slightly longer story: 

Around Spring 2019, a group of African engineers from MIT and Université Laval was discussing an idea for an NLP side project. The premise was at first seemingly simple. We wanted to understand whether one could compare two documents by comparing their summaries and detect documents from the same genre, or paraphrased documents (ie 2 news articles about the same facts) by first summarizing them and then applying a similarity metric on their summaries. But to do that, we had to compare summaries. “How hard could this be?” we thought. After all, word vectors, paragraph vectors, bi-directional LSTMs, sentence encoders and ,more recently, Transformers had made substantial amounts of progress in finding ways to condense sentences into vectors capturing their meaning. We assumed that we’d probably have to pick a pre-trained architecture, take its hidden state after passing the two summaries, compute cosine similarity between the hidden states and voilà! What we had not anticipated was that we would be designing a new paradigm to build evaluation metrics ourselves.

We first started by doing our homework, surveying the NLP literature to get up to speed and learn about the metrics currently used to assess summaries and machine translation. We stumbled upon BLEU and ROUGE, thinking it would solve our problems. However, upon closer look, we found out that BLEU and ROUGE, the metrics still used today (as of April 2020) to report the performance of machine translation and summarization systems at their core relied on n-grams overlap between candidates and reference sentences. Wait, what???  “That can’t be right” we thought.

We started asking ourselves “why hadn’t the similarity metrics evolved in almost 20 years?”. One of the things we discovered was that these metrics claimed to correlate with human judgement; however, the dataset used to measure this correlation was not readily available. 

It seemed that there was a growing disconnect between the amazing array of tools available given recent progress in NLP and the community practice of measuring quality of summaries. This disconnect is tied to the fact that there is not really a protocol to assess and evaluate metrics for text generation tasks. Why is that important?

Unlike traditional machine learning tasks such as classification and regression, text generation (i.e. machine translation, summarization, image captioning) is very nuanced and the gold standard is human evaluation. However, we can’t afford that for most models so instead, proxies in the form of machine learning models were designed to approximate human judgment of quality. A consequence of this unique setup is that until we get quasi-human judgment of quality (which we’re still far from) the metrics themselves have to be upgraded every couple years to reflect the dynamic progress of the field. However, that didn’t really happen and ,instead, we’re stuck with metrics used since 2002 and 2004. Given their popularity and simplicity, it has been virtually impossible to dislodge them. Some VCs would call BLEU and ROUGE “sticky” and a good example of network effect.

The question of coming up with a systematic protocol to benchmark metrics occupied our minds and was the subject of a paper we presented at the NeurIPS 2019 Document Intelligence workshop. On this one, we were joined by a brilliant NLP Researcher  from Boğaziçi University in Turkey who played a huge role in fleshing out this metric scorecard and laying the foundation for NUBIA.


After digging a bit, we also found out that other researchers raised red flags about the need to seriously vet evaluation metrics for language generation tasks. However, the community did not sufficiently pay attention to this problem and continues to this day to use ROUGE and BLEU while complaining about it. 

After jotting down thoughts and criteria on how to assess evaluation metrics, we realized that those criteria almost automatically gave a formulation for what a sound metric would be. After designing and testing it in the machine translation and image captioning space, we are excited to introduce NUBIA.

What is NUBIA, you may wonder? NUBIA is an AI system that does one particular job for now (sorry to disappoint AGI folks): given two sentences, a reference sentence A and a candidate sentence B, it outputs a score between 0 and 1 expressing how much it believes that A is interchangeable with B. 
It does it by comparing pairs of sentences along 3 axis : 
How semantically similar is the candidate sentence B to the reference sentence A ?
Is the candidate sentence in logical agreement (Reference implies Candidate), neutral (the candidate sentence can’t be deduced from the reference sentence either because they’re unrelated or the candidate sentence adds extra information not contained in the candidate sentence) or in contradiction with the reference sentence ?
Is the candidate sentence even well-written and free of grammatical mistakes ?
After scoring the sentence pairs along those dimensions, NUBIA computes a “holistic score” between 0 and 1 that captures its impression. To have a strong grasp of how NUBIA fares against BLEU and ROUGE, we provide some examples below.

INSERT video going through 5 examples here and discussing them. Also acknowledge some limitations. The goal here is to go to results and build up curiosity in the reader’s mind without actually explaining how NUBIA works.

The way we extract these scores and computes the holistic NUBIA score is beyond the scope of this blog but can be read about in our technical paper available on arxiv (link). In addition, we provide a self-contained implementation of NUBIA via this notebook here. The Github repo also has all the information required to reproduce the experiments from the paper. The web app is still a prototype but gives a flavor of how NUBIA’s analysis can be visualized and interpreted.

To be fair, as we embarked on this intellectual journey, we realized the more complicated reality that there were actually *many* criticisms of BLEU and ROUGE and very sophisticated proposals to replace them but most of them failed to get traction (check related work section of our paper). At the core, it’s because the NLP community hasn’t yet reached a consensus on an implementable proposal about how to evaluate new metrics and pick the successor to BLEU and ROUGE.

We think that this time, the timing is right because the NLP community itself is gearing up for an intense, and likely multi-year conversation on this topic as we transition from Twitter complaints/arguments to thoughtful proposals thoroughly discussed at an upcoming EMNLP workshop specifically dedicated to this question in November 2020. 
The rapid pace of progress in NLP is ,in some areas, both currently bottlenecked by language generation evaluation metrics and at the same time offering promising techniques with the potential to overcome these bottlenecks.

We’re excited to contribute to this conversation by releasing our work. 

If you’re a researcher investigating NLP and working on machine translation, summarization or image captioning, we’re excited to discuss how to incorporate NUBIA into your work. We also have ideas on potential applications and ways to improve and study the system but also felt confident enough that the 1+ year technical journey we’ve been on has yielded fruits mature enough to be shared with the wider AI community.

We’re still deeply thinking about the question of how to properly assess/diagnose evaluation metrics and design performant ones and will share more in the coming months. While we think NUBIA is a step in the right direction, we also know it is far from perfect.

Your feedback/questions would be most helpful after reading this blog post, skimming the paper and playing around with the notebook (be ready for a 10-15 minutes setup as the current model is 5GB). We kept the blog post intentionally short on technical details to convey the big picture/motivation behind this work and we suspect that the answers to many questions you may have are included in the paper or notebook/repo which provide an implementation of NUBIA.

Stay tuned! 

To add how NUBIA works and  link to the paper/repo/notebook/website


## Basic formatting

You can use *italics*, **bold**, `code font text`, and create [links](https://www.markdownguide.org/cheat-sheet/). Here's a footnote [^1]. Here's a horizontal rule:

---

## Lists

Here's a list:

- item 1
- item 2

And a numbered list:

1. item 1
1. item 2

## Boxes and stuff

> This is a quotation

{% include alert.html text="You can include alert boxes" %}

...and...

{% include info.html text="You can include info boxes" %}

## Images


## Code

General preformatted text:

    # Do a thing
    do_thing()

Python code and output:

```python
# Prints '2'
print(1+1)
```

    2

## Tables

| Column 1 | Column 2 |
|-|-|
| A thing | Another thing |

## Footnotes

[^1]: This is the footnote.

