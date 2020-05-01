from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os
import wget
import numpy as np
from fairseq.models.roberta import RobertaModel
from joblib import load

ROBERTA_STS_PATH = 'pretrained/roBERTa_STS'
ROBERTA_MNLI_PATH = 'pretrained/roBERTa_MNLI'
AGGREGATOR_DIR = 'pretrained/aggregators/'
AGGREGATOR_2015_2016 = \
    'pretrained/aggregators/nn_2015_2016_6_dim' \
    '.joblib'
AGGREGATOR_2015_2017 = \
    'pretrained/aggregators/nn_2015_2017_6_dim' \
    '.joblib'
AGGREGATOR_2015_2016_8_dim = \
    'pretrained/aggregators/nn_2015_2016_8_dim' \
    '.joblib'
AGGREGATOR_2015_2017_8_dim = \
    'pretrained/aggregators/nn_2015_2017_8_dim' \
    '.joblib'

ROBERTA_STS_URL = "https://nubia-nn.s3.amazonaws.com/" \
                  "neural-feature-extractors/checkpoint_best.pt"
ROBERTA_MNLI_URL = "https://nubia-nn.s3.amazonaws.com/" \
                   "neural-feature-extractors/model_mnli.pt"
AGGREGATOR_2015_2016_URL = "https://nubia-nn.s3.amazonaws.com/" \
                           "aggregators/nn_2015_2016_6_dim.joblib"
AGGREGATOR_2015_2017_URL = "https://nubia-nn.s3.amazonaws.com/" \
                           "aggregators/nn_2015_2017_6_dim.joblib"
AGGREGATOR_2015_2016_8_dim_URL = "https://nubia-nn.s3.amazonaws.com/" \
                           "aggregators/nn_2015_2016_8_dim.joblib"

AGGREGATOR_2015_2017_8_dim_URL = "https://nubia-nn.s3.amazonaws.com/" \
                           "aggregators/nn_2015_2017_8_dim.joblib"


class Nubia:
    def __init__(self):
        if not os.path.exists(AGGREGATOR_DIR):
            os.makedirs(AGGREGATOR_DIR)
        if not os.path.isfile(AGGREGATOR_2015_2016):
            print("Downloading aggregators from s3...")
            wget.download(AGGREGATOR_2015_2016_URL,
                                AGGREGATOR_2015_2016,
                                bar=self._download_progress_bar)
        if not os.path.isfile(AGGREGATOR_2015_2017):
            print("Downloading aggregators from s3...")
            wget.download(AGGREGATOR_2015_2017_URL,
                                AGGREGATOR_2015_2017,
                                bar=self._download_progress_bar)
        if not os.path.isfile(AGGREGATOR_2015_2016_8_dim):
            print("Downloading aggregators from s3...")
            wget.download(AGGREGATOR_2015_2016_8_dim_URL,
                                AGGREGATOR_2015_2016_8_dim,
                                bar=self._download_progress_bar)
        if not os.path.isfile(AGGREGATOR_2015_2017_8_dim):
            print("Downloading aggregators from s3...")
            wget.download(AGGREGATOR_2015_2017_8_dim_URL,
                                AGGREGATOR_2015_2017_8_dim,
                                bar=self._download_progress_bar)
        if not os.path.isfile(ROBERTA_STS_PATH + '/checkpoint_best.pt'):
            print("Downloading ROBERTA STS model from s3...")
            wget.download(ROBERTA_STS_URL, ROBERTA_STS_PATH +
                          '/checkpoint_best.pt',
                          bar=self._download_progress_bar)
        if not os.path.isfile(ROBERTA_MNLI_PATH + '/model_mnli.pt'):
            print("Downloading ROBERTA MNLI model from s3...")
            wget.download(ROBERTA_MNLI_URL, ROBERTA_MNLI_PATH +
                          '/model_mnli.pt', bar=self._download_progress_bar)

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
        self.agg_one_8_dim = load(AGGREGATOR_2015_2016_8_dim)
        self.agg_two_8_dim = load(AGGREGATOR_2015_2017_8_dim)

    @staticmethod
    def _download_progress_bar(current, total, width=80):
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

    def nubia(self, ref, hyp, get_features=False, six_dim=False,
              aggregator="agg_one"):
        sim = float(self._roberta_similarity(ref, hyp)[0])
        mnli_zero, mnli_one, mnli_two = self._roberta_mnli_all_values(ref, hyp)
        gpt_ref = self._gpt_score(ref)
        gpt_hyp = self._gpt_score(hyp)
        len_ref = len(ref.split(" "))
        len_hyp = len(hyp.split(" "))

        mnli_friendly = torch.nn.functional.softmax(
            torch.tensor([mnli_zero, mnli_one, mnli_two]), dim=0).tolist()

        neural_features_6_dim = np.array(
            [float(sim), float(mnli_zero), float(mnli_one), float(mnli_two),
             float(gpt_ref), float(gpt_hyp)])  # 6 Neural Features

        neural_features_8_dim = np.array(
            [float(sim), float(mnli_zero), float(mnli_one), float(mnli_two),
             float(gpt_ref), float(gpt_hyp), float(len_ref),
             float(len_hyp)])  # 8 Neural Features

        if aggregator == "agg_one":
            if six_dim:
                nubia_score = float(self.agg_one.predict(
                    neural_features_6_dim.reshape(1, -1))[0])
            else:
                nubia_score = float(self.agg_one_8_dim.predict(
                    neural_features_8_dim.reshape(1, -1))[0])
        else:
            if six_dim:
                nubia_score = float(self.agg_two.predict(
                    neural_features_6_dim.reshape(1, -1))[0])
            else:
                nubia_score = float(self.agg_two_8_dim.predict(
                    neural_features_8_dim.reshape(1, -1))[0])

        if get_features:
            return {"nubia_score": nubia_score, "features": {
                "semantic_relation": min(5.0, sim),
                "contradiction": mnli_friendly[0]*100,
                "irrelevancy": mnli_friendly[1]*100,
                "logical_agreement": mnli_friendly[2]*100,
                "grammar_ref": gpt_ref.item(),
                "grammar_hyp": gpt_hyp.item(),
            }
             }
        return nubia_score

    def score(self, ref, hyp, verbose=False, get_features=False,
              six_dim=False, aggregator="agg_one"):

        nubia = self.nubia(ref, hyp, get_features=True, six_dim=six_dim,
                           aggregator=aggregator)

        self_similarity = self.nubia(ref, ref,
                                     get_features=False, six_dim=six_dim,
                                     aggregator=aggregator)

        amplitude = abs(self_similarity) + 1
        difference = self_similarity - nubia["nubia_score"]

        calibrated = 1.0 - (float(difference) / float(amplitude))

        if calibrated > 0:
            calibrated = min(1.0, calibrated)
        else:
            calibrated = max(0.0, calibrated)

        if verbose:
            print("Semantic relation: " +
                  str(min(5.0, nubia["features"]["semantic_relation"])) +
                  '/5.0\n')
            print("Percent chance of contradiction: " +
                  str(nubia["features"]["contradiction"]) + "%\n")
            print("Percent chance of irrelevancy " +
                  str(nubia["features"]["irrelevancy"]) + "%\n")
            print("Percent chance of logical agreement " +
                  str(nubia["features"]["logical_agreement"]) + "%\n")
            print("Grammaticality score for reference sentence: "
                  + str(nubia["features"]["grammar_ref"]) + '\n')
            print("Grammaticality score for candidate sentence:  "
                  + str(nubia["features"]["grammar_hyp"]) + '\n\n')
            print("NUBIA score: " + str(calibrated) + "/1.0")

        nubia["nubia_score"] = calibrated

        if get_features:
            return nubia

        return calibrated


