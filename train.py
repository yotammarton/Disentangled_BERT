import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Modify if needed # TODO

import wandb
from functools import partial
import datasets
from transformers import EarlyStoppingCallback
from transformers import BertTokenizer, BertModel, BertConfig, BertLMHeadModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataset import load_preprocess_glucose_dataset
# Our own custom EncoderDecoderModel with Disentangled embeddings
from custom_modeling_encoder_decoder import EncoderDecoderModel

os.environ["WANDB_DISABLED"] = "true"  # set wandb automatic setup to false


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
                                split_embedding=split_embedding)

    return model


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


def train(split_embedding):
    """
    Train a custom EncoderDecoderModel on glucose dataset

    @param split_embedding: True: make the disentangled embedding split,
                            False: keep the model as is without the embedding split
    """
    print(f'[*] Training EncoderDecoderModel with split_embedding = {split_embedding}')
    n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    batch_size = 8
    print(f'[*] Using GPU(s): {os.environ["CUDA_VISIBLE_DEVICES"]} (n_gpus: {n_gpus}) and batch_size: {batch_size}')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    data = load_preprocess_glucose_dataset(batch_size=batch_size, tokenizer=tokenizer)
    train_data, val_data, test_data = data['train'], data['val'], data['test']

    model = create_encoder_decoder_model(split_embedding=split_embedding)
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.encoder.vocab_size

    # noinspection PyTypeChecker
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        # generation_max_length=128, # Might be needed to change
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,  # CUDA Only https://github.com/ThilinaRajapakse/simpletransformers/issues/646
        fp16_full_eval=True,
        output_dir="./",
        num_train_epochs=25,  # We have early stopping
        # save_steps=10,  # Only if we use save_strategy="steps"
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        # warmup_steps=2000, # Linear increasing of the lr
        gradient_accumulation_steps=1,  # 1 is the default,
        metric_for_best_model='rouge2_fmeasure',
        load_best_model_at_end=True
    )

    # metric can be changed
    # good article: https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213
    rouge = datasets.load_metric("rouge")

    compute_metrics = partial(_compute_metrics, rouge=rouge, tokenizer=tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=val_data,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    model.is_training = True
    trainer.train()
    model.is_training = False


if __name__ == "__main__":
    wandb.init(project="disentangled_bert", entity="yotammartin")
    split_embedding = True
    train(split_embedding=split_embedding)  # True = split embedding, False = normal model
