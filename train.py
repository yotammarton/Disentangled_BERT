import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertTokenizerFast
from model import EncoderDecoderModel
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig, BertLMHeadModel


def train_model(model):
    model.train()


def eval_model(model):
    model.eval()


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encoder
encoder_config = BertConfig(
    _name_or_path="bert-base-uncased",  # Uncomment for pre-trained weights
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
encoder = BertModel(config=encoder_config)

# Decoder
decoder_config = BertConfig(
    _name_or_path="bert-base-uncased",  # Uncomment for pre-trained weights
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

model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

"""Verify we got a random weights compared to a pre-trained model"""
# initialize Bert2Bert from pre-trained checkpoints
# org_model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased',
#                                                                 'bert-base-uncased')
# encoder_params_my = list(model.encoder.parameters())
# decoder_params_my = list(model.decoder.parameters())
# encoder_params_org = list(org_model.encoder.parameters())
# decoder_params_org = list(org_model.decoder.parameters())

optimizer = optim.Adam(model.parameters())

model.train()
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)
for i in range(100):
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
    loss, logits = outputs.loss, outputs.logits
    print(loss)
    loss.backward()
    optimizer.step()

# https://huggingface.co/blog/how-to-generate
# https://huggingface.co/transformers/main_classes/model.html#transformers.generation_utils.GenerationMixin.generate
generated = model.generate(input_ids, decoder_start_token_id=model.decoder.config.pad_token_id)
decoded_generated = tokenizer.decode(generated.view(-1).numpy())
