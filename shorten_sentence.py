# Sentence Simplifier using spaCy + benepar

import spacy
import benepar
import nltk
import re
from nltk.tree import ParentedTree
from anytree import AnyNode

# Download necessary models
nltk.download('punkt')

# Load spaCy with benepar
nlp = spacy.load("en_core_web_sm")
benepar.download('benepar_en3')
nlp.add_pipe("benepar", config={"model": "benepar_en3"})

# Util: Tokenize and pre-process conjunctions
def tokenize(sent):
    tokenized_sent = nltk.word_tokenize(sent)
    if ('If') in tokenized_sent and ('then') in tokenized_sent:
        tokenized_sent.remove('If')
        tokenized_sent.insert(tokenized_sent.index('then'),'and')
        tokenized_sent.remove('then')
    if ('because') in tokenized_sent:
        tokenized_sent.insert(tokenized_sent.index('because'),(','))
        tokenized_sent.insert(tokenized_sent.index('because')+1,(','))
        tokenized_sent.insert(tokenized_sent.index('because'),'and')
        tokenized_sent.remove('because')
    if ('while') in tokenized_sent:
        tokenized_sent.insert(tokenized_sent.index('while'),'and')
        tokenized_sent.remove('while')
    if ('which') in tokenized_sent:
        tokenized_sent.insert(tokenized_sent.index('which'),'and')
        tokenized_sent.remove('which')
    if ('or') in tokenized_sent:
        tokenized_sent.insert(tokenized_sent.index('or'),'and')
        tokenized_sent.remove('or')
    if ('who') in tokenized_sent:
        while (',') in tokenized_sent:
            tokenized_sent.insert(tokenized_sent.index(','),'and')
            tokenized_sent.remove(',')
        tokenized_sent.insert(tokenized_sent.index('who'),'and')
        tokenized_sent.remove('who')
    return tokenized_sent

# POS tagging using spaCy
def pos_tag(tokenized_sent):
    doc = nlp(" ".join(tokenized_sent))
    return [(token.text, token.tag_) for token in doc]

# Get constituency tree
def get_parse_tree(pos_tagged_tokens):
    sentence = " ".join([token for token, tag in pos_tagged_tokens])
    doc = nlp(sentence)
    for sent in doc.sents:
        return ParentedTree.fromstring(sent._.parse_string)

# Detect conjunction

def has_conj(tagged_sent):
    cc_list = [('and', 'CC'), ('but', 'CC')]
    for cc_pair in cc_list:
        if cc_pair in tagged_sent:
            return True
    return False

def split_needed(sent_list):
    for sent in sent_list:
        if has_conj(pos_tag(tokenize(sent))):
            return True
    return False

def rem_dup(list_):
    final = []
    for item in list_:
        if item not in final:
            final.append(item)
    return final

def split_util(sent):
    cc_list = [('and', 'CC'), ('but', 'CC')]
    for cc_pair in cc_list:
        if cc_pair in pos_tag(tokenize(sent)):
            return [s.strip() for s in sent.split(cc_pair[0]) if s.strip()]
    return [sent]

def simplify(sent):
    initial = [sent]
    final = []
    while split_needed(initial):
        final = []
        while initial:
            s = initial.pop(0)
            if split_needed([s]):
                for split_sent in reversed(split_util(s)):
                    final.append(split_sent)
            else:
                final.append(s)
        initial = final.copy()
    final = rem_dup(final)
    final = list(reversed(final))
    return final

# SBAR tree building (optional use)
def make_tree_sbar(tree, t, sent_list):
    if tree.label() not in sent_list:
        ttt = AnyNode(id=str(tree.label()), parent=t)
        for tt in tree:
            make_tree_sbar(tt, ttt, sent_list)
    else:
        AnyNode(id=str(tree), parent=t)

# Enhanced SBAR clause extraction
def extract_sbar_clauses(tree: ParentedTree):
    result = []

    def find_np_vp(t):
        subject, predicate = "", ""
        for child in t:
            if isinstance(child, ParentedTree):
                if child.label() == "NP":
                    subject = " ".join(child.leaves())
                elif child.label() == "VP":
                    predicate = " ".join(child.leaves())
        return subject, predicate

    for subtree in tree.subtrees():
        if subtree.label() == "S":
            subject, vp = find_np_vp(subtree)
            for child in subtree:
                if isinstance(child, ParentedTree) and child.label() == "SBAR":
                    for sub_s in child.subtrees(filter=lambda x: x.label() == "S"):
                        _, sb_vp = find_np_vp(sub_s)
                        if subject and sb_vp:
                            result.append(subject + " " + sb_vp)
            if subject and vp:
                result.append(subject + " " + vp)

    return result

# spaCy-based sentence tokenizer (handles abbreviations)
def spacy_sent_tokenize(paragraph):
    doc = nlp(paragraph)
    return [sent.text for sent in doc.sents]

# Simplify paragraph

def simplify_paragraph(paragraph):
    sentences = spacy_sent_tokenize(paragraph)
    results = []
    for sentence in sentences:
        tokenized_sent = tokenize(sentence)
        pos_tagged = pos_tag(tokenized_sent)
        tree = get_parse_tree(pos_tagged)

        if has_conj(pos_tagged):
            simplified = simplify(sentence)
            for i in simplified:
                results.append("Simple sentence: " + i.strip().capitalize() + ("." if not i.endswith(".") else ""))
        else:
            clauses = extract_sbar_clauses(tree)
            if clauses:
                for c in clauses:
                    results.append("Simple sentence: " + c.strip().capitalize() + ("." if not c.endswith(".") else ""))
            else:
                results.append("Simple sentence: " + sentence.strip())
    return results

# Example use
# paragraph = " Ms. Smith is a doctor and she is living at No.3 street. "
# simplified_sentences = simplify_paragraph(paragraph)
# for s in simplified_sentences:
    # print(s)
