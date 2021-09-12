import sys
import os
import json
import torch
from functools import partial
import datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertTokenizerFast
from model import EncoderDecoderModel  # A local copy of the source code with our modifications
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig, BertLMHeadModel, PretrainedConfig, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataset import load_preprocess_glucose_dataset


def create_encoder_decoder_model(split_embedding=False):
    # Encoder
    # We want to fine-tune (from_pretrained) because we want the BERT weights
    encoder_config = BertConfig(
        _name_or_path="bert-base-uncased",
        architectures=["BertForMaskedLM"],
        attention_probs_dropout_prob=0.1,
        gradient_checkpointing=False,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=768,
        initializer_range=0.02,
        intermediate_size=3072,
        layer_norm_eps=1e-12,
        max_position_embeddings=512,
        model_type="bert",
        num_attention_heads=12,
        num_hidden_layers=12,
        pad_token_id=0,
        position_embedding_type="absolute",
        transformers_version="4.4.2",
        type_vocab_size=2,
        use_cache=True,
        vocab_size=30522)
    encoder = BertModel.from_pretrained(pretrained_model_name_or_path="bert-base-uncased", config=encoder_config)

    # Decoder
    # We want to train from scratch because we want to make the split of the representation (trick)
    # and the decoder will need to learn the trick
    decoder_config = BertConfig(
        _name_or_path="bert-base-uncased",
        architectures=["BertForMaskedLM"],
        attention_probs_dropout_prob=0.1,
        gradient_checkpointing=False,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=768,
        initializer_range=0.02,
        intermediate_size=3072,
        layer_norm_eps=1e-12,
        max_position_embeddings=512,
        model_type="bert",
        num_attention_heads=12,
        num_hidden_layers=12,
        pad_token_id=0,
        position_embedding_type="absolute",
        transformers_version="4.4.2",
        type_vocab_size=2,
        use_cache=True,
        vocab_size=30522,
        is_decoder=True,
        add_cross_attention=True)
    decoder = BertLMHeadModel(config=decoder_config)

    model = EncoderDecoderModel(encoder=encoder, decoder=decoder, split_embedding=split_embedding)

    return model


def train_model(model):
    # Train the model for one epoch
    # TODO
    model.train()


def eval_model(model):
    model.eval()


def _compute_metrics(pred, rouge, tokenizer):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


def run():
    batch_size = 32
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    data = load_preprocess_glucose_dataset(batch_size=batch_size, tokenizer=tokenizer)
    train_data, test_data = data['train'], data['test']

    model = create_encoder_decoder_model(split_embedding=False)
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.encoder.vocab_size

    # noinspection PyTypeChecker
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # fp16=True, TODO CUDA Only
        output_dir="./",
        max_steps=40,  # TODO
        logging_steps=2,
        save_steps=10,
        eval_steps=4,
        # logging_steps=1000,
        # save_steps=500,
        # eval_steps=7500,
        # warmup_steps=2000,
        # save_total_limit=3,
    )

    # TODO can be changed
    #  good article: https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213
    rouge = datasets.load_metric("rouge")

    compute_metrics = partial(_compute_metrics, rouge=rouge, tokenizer=tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        # eval_dataset=???, TODO choose some
    )

    trainer.train()


if __name__ == "__main__":
    run()
    exit(777)

"""
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
Some extra code 
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*
"""

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = create_encoder_decoder_model(split_embedding=False)

"""Verify we got a random / pretrained weights compared to a pretrained model"""
# # initialize Bert2Bert from pre-trained checkpoints
# org_model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
#
# # look at the parameters weights. `org_model` is the pretrained ones. compare to desired `model`.
# encoder_params_my = list(model.encoder.parameters())
# decoder_params_my = list(model.decoder.parameters())
# encoder_params_org = list(org_model.encoder.parameters())
# decoder_params_org = list(org_model.decoder.parameters())

optimizer = optim.Adam(model.parameters())

model.train()
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)
for i in range(10):
    optimizer.zero_grad()
    # explanation of the next row
    # https://github.com/huggingface/transformers/issues/4517#issuecomment-636232107
    outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
    loss, logits = outputs.loss, outputs.logits
    print(loss)
    loss.backward()
    optimizer.step()

# https://huggingface.co/blog/how-to-generate
# https://huggingface.co/transformers/main_classes/model.html#transformers.generation_utils.GenerationMixin.generate
generated = model.generate(input_ids, decoder_start_token_id=model.decoder.config.pad_token_id)
decoded_generated = tokenizer.decode(generated.view(-1).numpy())
