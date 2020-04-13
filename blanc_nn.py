from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os
import wget
import numpy as np
from fairseq.models.roberta import RobertaModel
from joblib import load

ROBERTA_STS_PATH = 'pretrained/roBERTa_STS'
ROBERTA_MNLI_PATH = 'pretrained/roBERTa_MNLI'
AGGREGATOR_2015_2016 = \
    'pretrained/aggregators/nn_2015_2016_6_dim' \
    '.joblib'
AGGREGATOR_2015_2017 = \
    'pretrained/aggregators/nn_2015_2017_6_dim' \
    '.joblib'

ROBERTA_STS_URL = "https://blanc-nn.s3.amazonaws.com/" \
                  "neural-feature-extractors/checkpoint_best.pt"
ROBERTA_MNLI_URL = "https://blanc-nn.s3.amazonaws.com/" \
                   "neural-feature-extractors/model_mnli.pt"
AGGREGATOR_2015_2016_URL = "https://blanc-nn.s3.amazonaws.com/" \
                           "aggregators/nn_2015_2016_6_dim.joblib"
AGGREGATOR_2015_2017_URL = "https://blanc-nn.s3.amazonaws.com/" \
                           "aggregators/nn_2015_2017_6_dim.joblib"

class Blanc:
    def __init__(self):
        if not os.path.isfile(AGGREGATOR_2015_2016):
            print("Downloading aggregators from s3...")
            wget.download(AGGREGATOR_2015_2016_URL,
                                AGGREGATOR_2015_2016,
                                bar=self._download_prgoress_bar)
        if not os.path.isfile(AGGREGATOR_2015_2017):
            print("Downloading aggregators from s3...")
            wget.download(AGGREGATOR_2015_2017_URL,
                                AGGREGATOR_2015_2017,
                                bar=self._download_prgoress_bar)
        if not os.path.isfile(ROBERTA_STS_PATH + '/checkpoint_best.pt'):
            print("Downloading ROBERTA STS model from s3...")
            wget.download(ROBERTA_STS_URL, ROBERTA_STS_PATH +
                          '/checkpoint_best.pt',
                          bar=self._download_prgoress_bar)
        if not os.path.isfile(ROBERTA_MNLI_PATH + 'model_mnli.pt'):
            print("Downloading ROBERTA MNLI model from s3...")
            wget.download(ROBERTA_MNLI_URL, ROBERTA_MNLI_PATH +
                          '/model_mnli.pt', bar=self._download_prgoress_bar)

        self.roberta_STS = RobertaModel.from_pretrained(
            checkpoint_file='checkpoint_best.pt',
            model_name_or_path=ROBERTA_STS_PATH)
        self.roberta_STS.eval()

        self.roberta_MNLI = RobertaModel.from_pretrained(
            checkpoint_file='model_mnli.pt',
            model_name_or_path=ROBERTA_MNLI_PATH)
        self.roberta_MNLI.eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.agg_one = load(AGGREGATOR_2015_2016)
        self.agg_two = load(AGGREGATOR_2015_2017)

    @staticmethod
    def _download_prgoress_bar(current, total, width=80):
        print("Downloading: %d%% [%d / %d] bytes" % (
            current / total * 100, current, total))

    def _roberta_similarity(self, ref, hyp):
        tokens = self.roberta_STS.encode(ref, hyp)
        features = self.roberta_STS.extract_features(tokens)
        predicted_semantic_distance = 5.0 * \
            self.roberta_STS.model.classification_heads['sentence_classification_head'](features)
        return predicted_semantic_distance

    def _roberta_mnli_all_values(self, ref, hyp):
        tokens = self.roberta_MNLI.encode(ref, hyp)
        prediction = self.roberta_MNLI.predict('mnli', tokens)[0].\
            cpu().detach().numpy()
        return prediction

    def _gpt_score(self, text):
        tokenize_input = self.tokenizer.tokenize(text)
        tensor_input = torch.tensor([[self.tokenizer. eos_token_id] +
                                     self.tokenizer.convert_tokens_to_ids(
                                         tokenize_input)])
        with torch.no_grad():
            outputs = self.gpt_model(tensor_input, labels=tensor_input)
            loss, logits = outputs[:2]
        return loss

    def score(self, ref, hyp, get_features=False):
        sim = float(self._roberta_similarity(ref, hyp)[0])
        mnli_zero, mnli_one, mnli_two = self._roberta_mnli_all_values(ref, hyp)
        gpt_ref = self._gpt_score(ref)
        gpt_hyp = self._gpt_score(hyp)
        neural_features = np.array([float(sim), float(mnli_zero),
                                    float(mnli_one), float(mnli_two),
                                    float(gpt_ref), float(gpt_hyp)])
        blanc_score = self.agg_two.predict(neural_features.reshape(1, -1))
        if get_features:
            return {"blanc_score": blanc_score, "neural_features":
                    neural_features}
        return blanc_score[0]

