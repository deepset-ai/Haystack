"""
Script to perform data augmentation on a SQuAD like dataset to increase training data. It follows the approach oultined in the TinyBERT paper.
Usage:
    python augment_squad.py --squad_path <squad_path> --output_path <output_patn> \
        --multiplication_factor <multiplication_factor> --word_possibilities <word_possibilities> \
        --replace_probability <replace_probability>
Arguments:
    squad_path: Path to the input dataset. Must have the same structure as the official squad json.
    output_path: Path to the output dataset.
    multiplication_factor: Number of times to augment the dataset.
    word_possibilities: Number of possible words to replace a word with.
    replace_probability: Probability of replacing a word with a different word.
"""


import torch
from transformers import BertForMaskedLM, BertTokenizer
from copy import copy, deepcopy
from pathlib import Path
import requests
from zipfile import ZipFile
import numpy as np
import random
import argparse
import json
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def load_glove(glove_path: Path = Path("glove.txt"), vocab_size: int = 100_000):
    if not glove_path.exists():
        zip_path = glove_path.parent / (glove_path.name + ".zip")
        request = requests.get("https://nlp.stanford.edu/data/glove.42B.300d.zip", allow_redirects=True)
        with zip_path.open("wb") as downloaded_file:
            downloaded_file.write(request.content)
        with ZipFile(zip_path, "r") as zip_file:
            glove_file = zip_file.namelist()[0]
            with glove_path.open("wb") as g:
                g.write(zip_file.read(glove_file))

    word_id_mapping = {}
    id_word_mapping = {}
    vector_list = []
    with open(glove_path, "r") as f:
        for i, line in enumerate(f):
            if i == vocab_size:
                break
            split = line.split()
            word_id_mapping[split[0]] = i
            id_word_mapping[i] = split[0]
            vector_list.append(np.array([float(x) for x in split[1:]]))
    vectors = np.stack(vector_list)
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    print("loaded glove")
    return word_id_mapping, id_word_mapping, vectors

def tokenize_and_extract_words(text, tokenizer):
    words = tokenizer.basic_tokenizer.tokenize(text)

    subwords = [tokenizer.wordpiece_tokenizer.tokenize(word) for word in words]

    word_subword_mapping = {}

    j = 0
    for i, subwords_ in enumerate(subwords):
        if len(subwords_) == 1:
                word_subword_mapping[i] = j + 1
        j += len(subwords_)
    
    subwords = [subword for subwords_ in subwords for subword in subwords_] # flatten list of lists

    input_ids = tokenizer.convert_tokens_to_ids(subwords)
    input_ids.insert(0, tokenizer.cls_token_id)
    input_ids.append(tokenizer.sep_token_id)

    return input_ids, words, word_subword_mapping

def get_replacements(glove_word_id_mapping, glove_id_word_mapping, glove_vectors, model: BertForMaskedLM, tokenizer: BertTokenizer, text, word_possibilities=20, batch_size=16):
    input_ids, words, word_subword_mapping = tokenize_and_extract_words(text, tokenizer)

    inputs = []
    for word_index in word_subword_mapping:
        subword_index = word_subword_mapping[word_index]
        input_ids_ = copy(input_ids)
        input_ids_[subword_index] = tokenizer.mask_token_id
        inputs.append(input_ids_)
    
    with torch.no_grad():
        prediction_list = []
        while len(inputs) != 0:
            batch_list = inputs[:batch_size]
            batch = torch.tensor(batch_list)
            batch = batch.to("cuda:0")
            prediction_list.append(model(input_ids=batch)["logits"].cpu())
            inputs = inputs[batch_size:]
        predictions = torch.cat(prediction_list, dim=0)



    possible_words = []

    batch_index = 0
    for i, word in enumerate(words):
        if i in word_subword_mapping:
            subword_index = word_subword_mapping[i]
            logits = predictions[batch_index, subword_index]
            ranking = torch.argsort(logits, descending=True)[:word_possibilities]
            possible_words.append([word] + tokenizer.convert_ids_to_tokens(ranking))

            batch_index += 1
        elif word in glove_word_id_mapping:
            word_id = glove_word_id_mapping[word]
            glove_vector = glove_vectors[word_id]
            word_similarities = glove_vectors.dot(glove_vector)
            ranking = np.argsort(word_similarities)[-word_possibilities - 1:][::-1]
            possible_words.append([glove_id_word_mapping[id] for id in ranking])
        else:
            possible_words.append([word])

    return possible_words

def augment(word_id_mapping, id_word_mapping, vectors, model: BertForMaskedLM, tokenizer: BertTokenizer, text, multiplication_factor=20, word_possibilities=20, replace_probability=0.4):
    replacements = get_replacements(word_id_mapping, id_word_mapping, vectors, model, tokenizer, text, word_possibilities)
    new_texts = []
    for i in range(multiplication_factor):
        new_text = []
        for possible_words in replacements:
            if len(possible_words) == 1:
                new_text.append(possible_words[0])
                continue
            if random.random() < replace_probability:
                new_text.append(random.choice(possible_words[1:]))
            else:
                new_text.append(possible_words[0])
        new_texts.append(" ".join(new_text))
    return new_texts

def augment_squad(model: BertForMaskedLM, tokenizer: BertTokenizer, squad_path: Path, output_path: Path, glove_path: Path = Path("glove.txt"), multiplication_factor: int = 20, word_possibilities: int = 20, replace_probability: float = 0.4):
    word_id_mapping, id_word_mapping, vectors = load_glove(glove_path)

    with open(squad_path, "r") as f:
        squad = json.load(f)
        
    topics = []

    for topic in tqdm(squad["data"]):
        paragraphs = []
        for paragraph in topic["paragraphs"]:
            context = paragraph["context"]
            contexts = augment(word_id_mapping, id_word_mapping, vectors, model, tokenizer, context, multiplication_factor=multiplication_factor, word_possibilities=word_possibilities, replace_probability=replace_probability)
            qas = []
            for qa in paragraph["qas"]:
                question = qa["question"]
                question = augment(word_id_mapping, id_word_mapping, vectors, model, tokenizer, question, multiplication_factor=multiplication_factor, word_possibilities=word_possibilities, replace_probability=replace_probability)
                qas_ = []
                for question_ in question:
                    new_qa = deepcopy(qa)
                    new_qa["question"] = question_
                    qas_.append(new_qa)
                qas += qas_
            paragraphs_ = []
            for context in contexts:
                new_paragraph = deepcopy(paragraph)
                new_paragraph["context"] = context
                new_paragraph["qas"] = qas
                paragraphs_.append(new_paragraph)
            paragraphs += paragraphs_
        topic["paragraphs"] = paragraphs
        topics.append(topic)
    squad["topics"] = topics
    with open(output_path, "w") as f:
        json.dump(squad, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--squad_path", type=Path, required=True, help="Path to the squad json file")
    parser.add_argument("--output_path", type=Path, required=True, help="Path to save augmented dataaset")
    parser.add_argument("--multiplication_factor", type=int, default=4, help="Factor by which dataset size is multiplied")
    parser.add_argument("--word_possibilities", type=int, default=5, help="Number of possible words to choose from when replacing a word")
    parser.add_argument("--replace_probability", type=float, default=0.4, help="Probability of replacing a word")
    parser.add_argument("--glove_path", type=Path, default="glove.txt", help="Path to the glove file")

    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model = model.to("cuda:0")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    augment_squad(model, tokenizer, **vars(parser.parse_args()))