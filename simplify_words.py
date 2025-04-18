import torch
import spacy
from transformers import BertTokenizer, BertModel
from nltk.corpus import wordnet
from wordfreq import word_frequency
from scipy.spatial.distance import cosine

# Load spaCy and BERT
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

def get_wordnet_pos(spacy_pos):
    if spacy_pos.startswith("N"):
        return wordnet.NOUN
    elif spacy_pos.startswith("V"):
        return wordnet.VERB
    elif spacy_pos.startswith("J"):
        return wordnet.ADJ
    elif spacy_pos.startswith("R"):
        return wordnet.ADV
    return None

def get_synonyms(word, pos, top_n=20):
    wn_pos = get_wordnet_pos(pos)
    if wn_pos is None:
        return []

    # WordNet synonyms
    synsets = wordnet.synsets(word, pos=wn_pos)
    synonyms = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ")
            if synonym.lower() != word.lower() and synonym.isalpha():
                synonyms.add(synonym.lower())

    # Sort synonyms by word frequency and take top N
    sorted_syns = sorted(list(synonyms), key=lambda w: word_frequency(w, 'en'), reverse=True)
    return sorted_syns[:top_n]

def get_word_embedding(word, sentence):
    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    word_pieces = tokenizer.tokenize(word)
    for i in range(len(token_ids) - len(word_pieces)):
        if token_ids[i:i+len(word_pieces)] == word_pieces:
            return outputs.last_hidden_state[0][i].numpy()
    return None

def simplify_word(word, sentence, pos_tag, original_embedding, threshold=1e-5):
    candidates = get_synonyms(word, pos_tag, top_n=20)
    scored_candidates = []
    for cand in candidates:
        # if word_frequency(cand, 'en') < threshold:
            # continue
        new_sent = sentence.replace(word, cand)
        new_embedding = get_word_embedding(cand, new_sent)
        if new_embedding is None:
            continue
        sim = 1 - cosine(original_embedding, new_embedding)
        score = sim * word_frequency(cand, 'en')
        # score=sim
        scored_candidates.append((cand, score))
    if scored_candidates:
        return sorted(scored_candidates, key=lambda x: -x[1])[0][0]
    return word


def simplify_sentence(sentence):
    doc = nlp(sentence)
    tokens = [token.text for token in doc] 
    for i, token in enumerate(doc):
        if not token.is_alpha or token.is_stop:
            continue
        freq = word_frequency(token.text.lower(), 'en')
        if freq > 1e-5:
            continue
        orig_embed = get_word_embedding(token.text, sentence)
        if orig_embed is None:
            continue
        simple = simplify_word(token.text, sentence, token.tag_, orig_embed)
        if simple != token.text:
            tokens[i] = simple
    return " ".join(tokens)


# Example usage:
# text = "The professor elucidated the complex mechanism."
# print(text)
# print(simplify_sentence(text))
