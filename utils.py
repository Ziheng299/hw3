import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.
def typos(words, typo_prob):
    for i, word in enumerate(words):
        if random.uniform(0, 1) < typo_prob and len(word) > 3:
            typo_type = random.choice(["swap", "remove", "insert"])
            mid_pos = len(word) // 2
            if typo_type == "swap":
                pos = random.choice(list(range(len(word) - 1)))
                word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
            elif typo_type == "remove":
                pos = random.choice([mid_pos - 1, mid_pos, mid_pos + 1])
                word = word[:pos] + word[pos + 1:]
            elif typo_type == "insert":
                pos = random.choice([mid_pos - 1, mid_pos, mid_pos + 1])
                word = word[:pos] + word[pos] + word[pos:]
            words[i] = word
    return words

### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.
def synonym_replacement(words, synonym_prob):
    for i, word in enumerate(words):
        if random.uniform(0, 1) < synonym_prob:
            synonyms = set()
            for syn in wordnet.synsets(word):
                synonyms.add(syn.name().split('.')[0]) 
            # check if there exist synonyms
            if synonyms:
                words[i] = random.choice(list(synonyms))
    return words

# Adding emphasis words before adjectives or adverbs
def add_emphasis_words(words, emphasis_prob):
    emphasis_words = ["really", "very", "quite", "extremely", "super"]
    pos_tags = nltk.pos_tag(words, tagset='universal')
    for i, (word, pos) in enumerate(pos_tags):
        if pos in {"ADJ", "ADV"}:
            if random.uniform(0, 1) < emphasis_prob:
                emphasis = random.choice(emphasis_words)
                words[i] = emphasis + " " + word
    return words
    
def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    words = word_tokenize(example["text"])

    typo_prob = 0.25
    synonym_prob = 0.2
    emphasis_prob = 0.15

    words = synonym_replacement(words)
    words = typos(words)
    words = add_emphasis_words(words)

    example["text"] = TreebankWordDetokenizer().detokenize(words)
    ##### YOUR CODE ENDS HERE ######

    return example
