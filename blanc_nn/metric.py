from pytorch_transformers import *
import torch
import numpy as np
from fairseq.models.roberta import RobertaModel
from joblib import load


roberta_STS = RobertaModel.from_pretrained(
    checkpoint_file='checkpoint_best.pt',
    model_name_or_path='blanc_nn/pretrained/roBERTa_STS')

roberta_STS.eval()

roberta_MNLI = RobertaModel.from_pretrained(
    checkpoint_file='model_mnli.pt',
    model_name_or_path='blanc_nn/pretrained/roBERTa_MNLI')

roberta_MNLI.eval()


def roberta_similarity(sent1,sent2):
    tokens = roberta_STS.encode(sent1, sent2)
    features = roberta_STS.extract_features(tokens)
    predicted_semantic_distance = 5.0 * roberta_STS.model.classification_heads['sentence_classification_head'](features)
    return predicted_semantic_distance


def roberta_mnli_all_values(sent1,sent2):
    tokens = roberta_MNLI.encode(sent1, sent2)
    prediction = roberta_MNLI.predict('mnli', tokens)[0].cpu().detach().numpy()
    return prediction


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


def gpt_score(text):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    tokenize_input = tokenizer.tokenize(text)
    #50256 is the token_id for <|endoftext|>
    tensor_input = torch.tensor([ [tokenizer. eos_token_id]  +  tokenizer.convert_tokens_to_ids(tokenize_input)])
    with torch.no_grad():
        outputs = model(tensor_input, labels=tensor_input)
        loss, logits = outputs[:2]
    return loss


aggregator_2015_2016 = 'blanc_nn/pretrained/aggregators/nn_2015_2016_6_dim' \
                       '.joblib'
aggregator_2015_2017 = 'blanc_nn/pretrained/aggregators/nn_2015_2017_6_dim' \
                       '.joblib'

agg_one = load(aggregator_2015_2016)
agg_two = load(aggregator_2015_2017)

def blanc_nn(ref,hyp):
    sim = float(roberta_similarity(ref,hyp)[0])
    mnli_zero, mnli_one, mnli_two = roberta_mnli_all_values(ref, hyp)
    gpt_ref = gpt_score(ref)
    gpt_hyp = gpt_score(hyp)
    neural_features = np.array([float(sim),float(mnli_zero), float(mnli_one), float(mnli_two), float(gpt_ref), float(gpt_hyp)]) # 6 Neural Features
    blanc_score = agg_two.predict(neural_features.reshape(1,-1))
    print(neural_features)
    return blanc_score


print(blanc_nn("I've observed two ladies walking", "I've observed two ladies walking")[0])
