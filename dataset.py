from datasets import load_dataset
from functools import partial
from transformers import BertTokenizer


def _process_data_to_model_inputs(batch, tokenizer):
    # https://colab.research.google.com/drive/1WIk2bxglElfZewOHboPFNj8H44_VAyKE?usp=sharing
    # tokenize the texts
    tokenized = tokenizer(batch["story"], padding="max_length", truncation=True)

    batch["input_ids"] = tokenized.input_ids
    batch["attention_mask"] = tokenized.attention_mask
    batch["decoder_input_ids"] = tokenized.input_ids.copy()
    batch["decoder_attention_mask"] = tokenized.attention_mask.copy()
    batch["labels"] = tokenized.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in
                       batch["labels"]]

    return batch


def load_preprocess_glucose_dataset(batch_size, tokenizer):
    dataset = load_dataset("glucose")

    train_data = dataset['train']
    test_data = dataset['test']

    process_data_to_model_inputs = partial(_process_data_to_model_inputs, tokenizer=tokenizer)

    train_data = train_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=[col for col in train_data.column_names if col != 'story']
    )

    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )

    test_data = test_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=[col for col in test_data.column_names if col != 'story']
    )

    test_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )

    return {'train': train_data, 'test': test_data}
