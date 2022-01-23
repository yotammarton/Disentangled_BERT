import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Modify if needed TODO

from typing import Union
import datasets
import pandas as pd
import torch
from transformers import BertTokenizer
from custom_modeling_encoder_decoder import EncoderDecoderModel
from datasets import load_dataset
import numpy as np


def generate_sentences(sentences: Union[datasets.arrow_dataset.Dataset, list],
                       model) -> pd.DataFrame:
    """
    Generate sentences for given original sentences (we try to generate the same sentence exactly)
    @param sentences: Dataset with sentences or list of strings
    @param model: EncoderDecoderModel
    @return: pd.DataFrame with original sentences and generated sentences by the EncoderDecoderModel
    """
    if type(sentences) == datasets.arrow_dataset.Dataset:
        sentences = sentences['story']

    original_sentences, generated_sentences = list(), list()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for sentence in sentences:
        input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0).to("cuda")
        generated = model.generate(input_ids,
                                   decoder_start_token_id=model.decoder.config.pad_token_id,
                                   max_length=300)
        decoded_generated = tokenizer.decode(generated.view(-1).cpu().numpy())

        original_sentences.append(sentence)
        generated_sentences.append(decoded_generated)

    return pd.DataFrame(dict(original_sentences=original_sentences,
                             generated_sentences=generated_sentences))


if __name__ == '__main__':
    # Check generated sentences on the val or test set
    val_data = load_dataset("glucose", split="train[:5%]")
    test_data = load_dataset("glucose", split="test")
    dataset = val_data
    # Sample a subset of sentences from test set
    n = 10  # number of sentences to sample
    indices = np.random.choice(list(range(len(dataset))), size=n)
    sample = dataset.select(indices)
    # Load model
    checkpoint = "checkpoint-15568"  # best checkpoint with split_embedding=True (disentangled trick)
    model = EncoderDecoderModel.from_pretrained(checkpoint).to("cuda")
    # Put the model in eval mode
    model.is_training = False

    # Generate
    all_results_df = pd.DataFrame()
    model.split_embedding = True  # Even if saved model trained with split_embedding=True it is saved with False value
    # we split (to two parts) the hidden embeddings and multiply each of the two parts by `embed1` and `embed2`
    for embed1, embed2 in [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]:
        model.embed1, model.embed2 = embed1, embed2
        results_df = generate_sentences(sentences=sample, model=model)
        results_df['embed1'] = embed1
        results_df['embed2'] = embed2
        all_results_df = pd.concat([all_results_df, results_df])
    all_results_df.to_csv(f'results_df_{checkpoint}.csv', index=False)
