import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"  # TODO modify if needed

import wandb
import sys
import json
import torch
from functools import partial
import datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertTokenizerFast
# from model import EncoderDecoderModel  # A local copy of the source code with our modifications # TODO uncomment
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig, BertLMHeadModel, PretrainedConfig, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments, EncoderDecoderModel  # TODO remove original EncoderDecoderModel
from dataset import load_preprocess_glucose_dataset, load_preprocess_cnn_dataset  # TODO cnn del

os.environ["WANDB_DISABLED"] = "true"  # TODO wandb


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
        transformers_version="4.13.0.dev0",
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
        transformers_version="4.13.0.dev0",
        type_vocab_size=2,
        use_cache=True,
        vocab_size=30522,
        is_decoder=True,
        add_cross_attention=True)
    decoder = BertLMHeadModel(config=decoder_config)

    model = EncoderDecoderModel(encoder=encoder,
                                decoder=decoder,
                                # split_embedding=split_embedding  # TODO only in our model from `model.py`
                                )
    return model


def train_model(model):
    # Train the model for one epoch
    # TODO Implement
    model.train()


def eval_model(model):
    # TODO Implement
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
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    batch_size = 96
    per_device_train_batch_size=48
    ml-srv3              Mon Nov 15 00:57:27 2021  460.67
    [0] Tesla M40        | 26'C,   0 % |     0 / 11448 MB |
    [1] Tesla M40        | 25'C,   0 % |     0 / 11448 MB |
    [2] Tesla M40        | 62'C,  99 % | 11372 / 11448 MB | yotamm:python/659(11367M)
    [3] Tesla M40        | 64'C,  98 % |  8922 / 11448 MB | yotamm:python/659(8917M)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    batch_size = 32
    per_device_train_batch_size=32
    ml-srv3              Wed Nov 17 17:45:32 2021  460.67
    [0] Tesla M40        | 52'C,  68 % |  9474 / 11448 MB | yotamm:python/2101(9469M)
    [1] Tesla M40        | 52'C,  84 % |  6560 / 11448 MB | yotamm:python/2101(6555M)
    [2] Tesla M40        | 50'C,  90 % |  6560 / 11448 MB | yotamm:python/2101(6555M)
    [3] Tesla M40        | 52'C, 100 % |  6560 / 11448 MB | yotamm:python/2101(6555M)
    """
    n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    batch_size = 8
    print(f'[*] Using GPU(s): {os.environ["CUDA_VISIBLE_DEVICES"]} (n_gpus: {n_gpus}) and batch_size: {batch_size}')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    data = load_preprocess_glucose_dataset(batch_size=batch_size, tokenizer=tokenizer)
    train_data, val_data, test_data = data['train'], data['val'], data['test']

    model = create_encoder_decoder_model(split_embedding=False)
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.encoder.vocab_size

    # noinspection PyTypeChecker
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,  # TODO buggy
        # generation_max_length=128, # TODO needed?
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,  # TODO CUDA Only https://github.com/ThilinaRajapakse/simpletransformers/issues/646
        fp16_full_eval=True,  # TODO needed?
        output_dir="./",
        num_train_epochs=3,  # TODO just for checking everything works
        # save_steps=10,  # TODO if we use save_strategy="steps"
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        # warmup_steps=2000, # TODO linear increasing of the lr
        gradient_accumulation_steps=1  # TODO 1 is the default
    )

    # TODO metric can be changed
    #  good article: https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213
    rouge = datasets.load_metric("rouge")

    compute_metrics = partial(_compute_metrics, rouge=rouge, tokenizer=tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=val_data,
    )

    trainer.train()


if __name__ == "__main__":
    wandb.init(project="disentangled_bert", entity="yotammartin")  # TODO remove in the end
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
org_model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
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
