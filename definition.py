from wsd import get_wordnet_pos, get_list_of_synsets, word_sense_disambigution_bert, word_sense_disambiguation_GlossBERT
from nltk.wsd import lesk
from nltk.corpus import wordnet
import nltk
import xml.etree.ElementTree as ET
import os

nltk.download('universal_tagset', quiet=True)

def get_definition_for_target_word_in_sentence(sentence, target_word, location, algo="lesk"):
    """
    Get the most similar synset for the target word in the sentence based on sentence embedding.
    """

    ##################################################################################
    # Step 1: Tokenize the sentence and get the pos_tag
    ##################################################################################

    tokenized_sentence = nltk.word_tokenize(sentence.lower())
    tokenized_sentence = nltk.tag.pos_tag(tokenized_sentence, tagset='universal')

    ##################################################################################
    # Step 2: Get the POS tag for the target word
    ##################################################################################

    target_word_pos_tag = None
    if location >= 0 and location < len(tokenized_sentence):
        token = tokenized_sentence[location]
        if token[0].lower() == target_word.lower():
            target_word_pos_tag = token[1]
        else:
            print(sentence)
            print(f"Target word '{target_word}' not found in the sentence.")
            return None
    else:
        print(sentence)
        print(f"Location {location} is out of bounds for the tokenized sentence.")
        return None

    ##################################################################################
    # Step 3: Lemmatize the target word
    ##################################################################################

    lemmatized_target_word = None
    try:
        lemmatized_target_word = nltk.stem.WordNetLemmatizer().lemmatize(target_word, target_word_pos_tag[0])
    except Exception as e:
        lemmatized_target_word = nltk.stem.WordNetLemmatizer().lemmatize(target_word)

    ##################################################################################
    # Step 4: Apply word sense disambiguation to get the most similar synset
    ##################################################################################

    if algo == "lesk":
        definition = lesk(sentence.split(), target_word, pos=get_wordnet_pos(target_word_pos_tag))
    elif algo == "bert":
        definition = word_sense_disambigution_bert(sentence, lemmatized_target_word, pos_tag=target_word_pos_tag, aug=False)
    elif algo == "bert_aug":
        definition = word_sense_disambigution_bert(sentence, lemmatized_target_word, pos_tag=target_word_pos_tag, aug=True)
    elif algo == "glossbert":
        definition = word_sense_disambiguation_GlossBERT(sentence, target_word, lemmatized_target_word, pos_tag=target_word_pos_tag, aug=False)

    return definition


def demo():

    examples = [
        {"sentence": "I went to the bank to deposit some money into my bank account.", "target_word": "bank", "location": 4, "pos": wordnet.NOUN},
        {"sentence": "I went to the bank to deposit some money into my bank account.", "target_word": "deposit", "location": 6, "pos": wordnet.VERB},
        {"sentence": "I went to the bank to deposit some money into my bank account.", "target_word": "account", "location": 12, "pos": wordnet.NOUN},
        {"sentence": "The fox jumped over the lazy dog.", "target_word": "fox", "location": 1, "pos": wordnet.NOUN},
        {"sentence": "The fox jumped over the lazy dog.", "target_word": "jumped", "location": 2, "pos": wordnet.VERB},
    ]

    for example in examples:
        sentence = example["sentence"]
        target_word = example["target_word"]
        location = example["location"]
        pos = example["pos"]

        print(f"Example: {sentence}")
        print(f"Target word: '{target_word}' at location {location}")

        # List of synsets for the target word
        synsets = get_list_of_synsets(target_word, pos)
        # Print list of definitions for the target word
        print("Definitions:")
        for synset in synsets:
            print(f"- {synset} - {synset.definition()}")

        # Using Lesk algorithm
        definition = get_definition_for_target_word_in_sentence(sentence, target_word, location=location, algo="lesk")
        if definition:
            print(f"Lesk Definition: {definition.definition()}")
        else:
            print("Lesk Definition: None")

        # Using BERT-based method
        definition = get_definition_for_target_word_in_sentence(sentence, target_word, location=location, algo="bert")
        if definition:
            print(f"BERT Definition: {definition.definition()}")
        else:
            print("BERT Definition: None")

        # Using BERT-based method with augmentation
        definition = get_definition_for_target_word_in_sentence(sentence, target_word, location=location, algo="bert_aug")
        if definition:
            print(f"BERT Augmented Definition: {definition.definition()}")
        else:
            print("BERT Augmented Definition: None")

        # Using GlossBERT method
        definition = get_definition_for_target_word_in_sentence(sentence, target_word, location=location, algo="glossbert")
        if definition:
            print(f"GlossBERT Definition: {definition.definition()}")
        else:
            print("GlossBERT Definition: None")

        print("-" * 80)

def evaluate(xml_path, key_path, output_path, algo="bert"):
    
    def form_sentence(sentence):
        sentence_text = ""
        is_first = True
        start = False
        next_no_space = False
        for elem in sentence:
            if is_first:
                if elem.text == "``":
                    sentence_text += '"'
                    start = True
                    next_no_space = True
                else:
                    sentence_text += elem.text
                is_first = False
            elif elem.text == "``":
                if start:
                    sentence_text += '"'
                    start = False
                    next_no_space = True
                else:
                    sentence_text += ' "'
                    start = True
                    next_no_space = True
            elif next_no_space:
                sentence_text += elem.text
                next_no_space = False
            elif elem.attrib.get("pos") == "." and elem.text != "--":
                sentence_text += elem.text
            elif elem.text == "n't":
                sentence_text += elem.text
            elif elem.text.startswith("'") or elem.attrib.get("lemma").startswith("'"):
                sentence_text += elem.text
            else:
                sentence_text += ' ' + elem.text

        if '. "' in sentence_text:
            sentence_text = sentence_text.replace('. "', ' . " ')

        return sentence_text

    # Load the XML file
    tree = ET.parse(xml_path)

    with open(output_path, "w") as f:
        for sentence in tree.iter('sentence'):
            sentence_text = form_sentence(sentence)
            for elem in sentence:
                if elem.tag == "instance":
                    if algo == "bert":
                        definition = word_sense_disambigution_bert(sentence_text, elem.attrib.get("lemma"), pos_tag=elem.attrib.get("pos"), aug=False)
                    elif algo == "bert_aug":
                        definition = word_sense_disambigution_bert(sentence_text, elem.attrib.get("lemma"), pos_tag=elem.attrib.get("pos"), aug=True)
                    elif algo == "lesk":
                        definition = lesk(sentence_text.split(), elem.attrib.get("lemma"), pos=get_wordnet_pos(elem.attrib.get("pos")))
                    elif algo == "glossbert":
                        definition = word_sense_disambiguation_GlossBERT(sentence_text, elem.text, elem.attrib.get("lemma"), pos_tag=elem.attrib.get("pos"), aug=False)
                        
                    text = elem.attrib.get('id') + ' '
                    if definition:
                        text += definition.lemmas()[0].key() + '\n'
                    else:
                        text += f"None for {elem.text} \n"
                    f.write(text)

    # run java command to run the evaluation script
    os.system("java -cp .\\WSD_Unified_Evaluation_Datasets\\WSD_Unified_Evaluation_Datasets Scorer " + key_path + " " + output_path)

if __name__ == "__main__":

    demo()

    datasets = {
        "Senseval2": ".\\WSD_Unified_Evaluation_Datasets\\WSD_Unified_Evaluation_Datasets\\senseval2",
        "Senseval3": ".\\WSD_Unified_Evaluation_Datasets\\WSD_Unified_Evaluation_Datasets\\senseval3",
        "semeval2007": ".\\WSD_Unified_Evaluation_Datasets\\WSD_Unified_Evaluation_Datasets\\semeval2007",
        "semeval2013": ".\\WSD_Unified_Evaluation_Datasets\\WSD_Unified_Evaluation_Datasets\\semeval2013",
        "semeval2015": ".\\WSD_Unified_Evaluation_Datasets\\WSD_Unified_Evaluation_Datasets\\semeval2015",
        "ALL": ".\\WSD_Unified_Evaluation_Datasets\\WSD_Unified_Evaluation_Datasets\\ALL",
    }

    for name, path in datasets.items():
        print("-" * 80)
        print(f"Evaluating {name}")
        xml_path = os.path.join(path, f"{name.lower()}.data.xml")
        key_path = os.path.join(path, f"{name.lower()}.gold.key.txt")
        output_path = os.path.join(path, "output.key.txt")
        print("Lesk:")
        evaluate(xml_path, key_path, output_path, algo="lesk")
        print("BERT:")
        evaluate(xml_path, key_path, output_path, algo="bert")
        print("BERT with Augmentation:")
        evaluate(xml_path, key_path, output_path, algo="bert_aug")
        print("GlossBERT:")
        evaluate(xml_path, key_path, output_path, algo="glossbert")

