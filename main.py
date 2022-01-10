"""
The module contains some early stages code checking
"""

if __name__ == '__main__':
    """Example - BERT"""
    # model_name = 'bert-base-uncased'
    # sentence = 'an example of a possible sentence'
    # tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
    # encoding_model = transformers.BertModel.from_pretrained(model_name)
    #
    # with torch.no_grad():
    #     encoded_sequence = tokenizer.encode(sentence, truncation=True, return_tensors='pt',
    #                                         max_length=encoding_model.config.max_position_embeddings)
    #     decoded_sequence = tokenizer.decode(encoded_sequence.view(-1).numpy())
    #     out = encoding_model.forward(encoded_sequence)
    #     sentence_encoding = out.pooler_output  # [1, 768]
    #     sentence_last_hidden_state = out.pooler_output  # [1, len(decoded_sequence), 768] # hidden for every token

    """EncoderDecoderModel"""
    # https://huggingface.co/transformers/model_doc/encoderdecoder.html
    from transformers import BertTokenizer
    from transformers.models.encoder_decoder import EncoderDecoderModel
    import torch

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # initialize Bert2Bert from pre-trained checkpoints
    model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased',
                                                                'bert-base-uncased')

    # forward
    # Batch size 1
    input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)
    outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)

    # training
    outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
    loss, logits = outputs.loss, outputs.logits

    # # save and load from pretrained
    model.save_pretrained("bert2bert")
    model = EncoderDecoderModel.from_pretrained("bert2bert")

    # generation
    generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)
    decoded_generated = tokenizer.decode(generated.view(-1).numpy())
