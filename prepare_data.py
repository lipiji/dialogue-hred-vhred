# Preprocess cornell movie dialogs dataset
import os
from multiprocessing import Pool
import argparse
import pickle
import random
from pathlib import Path
from tqdm import tqdm
from utils import Vocab, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

project_dir = Path(__file__).resolve().parent
datasets_dir = project_dir.joinpath('./data/')

def load_conversations(fileName, spliter="</s>"):
    conversations = []
    with open(fileName, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fs = line.split("\t")
            if len(fs) != 2:
                print("error line", line)
            context, response = fs[0].strip(), fs[1].strip()
            utterances = context.split(spliter)
            conversation = []
            for utterance in utterances:
                conversation.append(utterance.split())
            conversation.append(response.split())
            conversations.append(conversation)
    return conversations

def tokenize_conversation(lines):
    sentence_list = [tokenizer(line['text']) for line in lines]
    return sentence_list

def pad_sentences(conversations, max_sentence_length=40, max_conversation_length=10):
    def pad_tokens(tokens, max_sentence_length=max_sentence_length):
        n_valid_tokens = len(tokens)
        if n_valid_tokens > max_sentence_length - 1:
            tokens = tokens[:max_sentence_length - 1]
        n_pad = max_sentence_length - n_valid_tokens - 1
        tokens = tokens + [EOS_TOKEN] + [PAD_TOKEN] * n_pad
        return tokens

    def pad_conversation(conversation):
        conversation = [pad_tokens(sentence) for sentence in conversation]
        return conversation

    all_padded_sentences = []
    all_sentence_length = []

    for conversation in conversations:
        if len(conversation) > max_conversation_length:
            conversation.reverse()
            conversation = conversation[:max_conversation_length]
            conversation.reverse() # the last n utterances
        sentence_length = [min(len(sentence) + 1, max_sentence_length) # +1 for EOS token
                           for sentence in conversation]
        all_sentence_length.append(sentence_length)

        sentences = pad_conversation(conversation)
        all_padded_sentences.append(sentences)

    sentences = all_padded_sentences
    sentence_length = all_sentence_length
    return sentences, sentence_length


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Maximum valid length of sentence
    # => SOS/EOS will surround sentence (EOS for source / SOS for target)
    # => maximum length of tensor = max_sentence_length + 1
    parser.add_argument('-s', '--max_sentence_length', type=int, default=40)
    parser.add_argument('-c', '--max_conversation_length', type=int, default=10)

    # Vocabulary
    parser.add_argument('--max_vocab_size', type=int, default=50000)
    parser.add_argument('--min_vocab_frequency', type=int, default=1)

    args = parser.parse_args()

    max_sent_len = args.max_sentence_length
    max_conv_len = args.max_conversation_length
    max_vocab_size = args.max_vocab_size
    min_freq = args.min_vocab_frequency

    print("Loading conversations...")
    train = load_conversations(datasets_dir.joinpath("train.txt"))
    valid = load_conversations(datasets_dir.joinpath("dev.txt"))
    test = load_conversations(datasets_dir.joinpath("test.txt"))

    print("#train=%d, #val=%d, #test=%d"%(len(train), len(valid), len(test)))
    def to_pickle(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    vocab = Vocab(lang="zh")
    for split_type, conversations in [('train', train), ('valid', valid), ('test', test)]:
        print(f'Processing {split_type} dataset...')
        split_data_dir = datasets_dir.joinpath(split_type)
        split_data_dir.mkdir(exist_ok=True)
        conversation_length = [min(len(conv), max_conv_len)
                               for conv in conversations]

        sentences, sentence_length = pad_sentences(
            conversations,
            max_sentence_length=max_sent_len,
            max_conversation_length=max_conv_len)

        print('Saving preprocessed data at', split_data_dir)
        to_pickle(conversation_length, split_data_dir.joinpath('conversation_length.pkl'))
        to_pickle(sentences, split_data_dir.joinpath('sentences.pkl'))
        to_pickle(sentence_length, split_data_dir.joinpath('sentence_length.pkl'))

        if split_type != 'test':
            print('Save Vocabulary...')
            vocab.add_dataframe(conversations)
            vocab.update(max_size=max_vocab_size, min_freq=min_freq)

            print('Vocabulary size: ', len(vocab))
            vocab.pickle(datasets_dir.joinpath('word2id.pkl'), datasets_dir.joinpath('id2word.pkl'))

    print('Done!')
