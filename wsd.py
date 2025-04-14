import nltk
from nltk.corpus import wordnet
import torch
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity

# Ensure that the WordNet data is downloaded
nltk.download('wordnet', quiet=True)

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

tokenizerGlossBert = AutoTokenizer.from_pretrained("kanishka/GlossBERT")
modelGlossBert = BertForSequenceClassification.from_pretrained("kanishka/GlossBERT")

def get_wordnet_pos(tag):
    """
    Convert Universal POS tags to WordNet POS tags: ADJ, ADJ_SAT, ADV, NOUN or VERB.
    """
    if tag == 'ADJ':
        return wordnet.ADJ
    elif tag == 'ADV':
        return wordnet.ADV
    elif tag == "VERB":
        return wordnet.VERB
    elif tag == 'NOUN':
        return wordnet.NOUN
    else:
        return None

def get_list_of_synsets(word, pos=None):
    """
    Get the list of synsets for a word from WordNet.
    """
    # Get the synsets for the word
    if pos is None:
        synsets = wordnet.synsets(word)
    else:
        synsets = wordnet.synsets(word, pos=pos)
    
    return synsets

def get_sentence_embedding(sentence):
    """
    Get the sentence embedding using BERT with GPU support.
    """
    # Add special tokens for BERT
    sentence = "[CLS] " + sentence + " [SEP]"

    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

    # Move inputs to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model.to(device)

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Sum last 4 hidden states
    hidden_states = outputs.hidden_states
    last_hidden_state = torch.stack(hidden_states[-1:], dim=0).sum(dim=0)

    # Return the mean of the last hidden state as the sentence embedding
    return last_hidden_state.mean(dim=1).cpu()  # Move result back to CPU

def word_sense_disambigution_bert(sentence, target_word, aug=False, pos_tag=None):

    ##################################################################################
    # Step 1: Get the list of synsets for the target word
    ##################################################################################

    synsets = get_list_of_synsets(target_word, get_wordnet_pos(pos_tag))
    if len(synsets) == 0:
        print(sentence)
        print(f"No synset found for '{target_word}' '{target_word}' '{pos_tag}'.")
        return None
    
    ##################################################################################
    # Step 2: Get the sentence embedding
    ##################################################################################

    sentence_embedding = get_sentence_embedding(sentence)

    ##################################################################################
    # Step 3: Get the sentence embeddings for the definition of each synset
    ##################################################################################

    definition_embeddings = []
    for synset in synsets:
        if aug:
            examples = " For Example: " + "; ".join(synset.examples()) if synset.examples() else ""
        else:
            examples = ""
        text = synset.definition() + examples
        definition_embedding = get_sentence_embedding(text)
        definition_embeddings.append(definition_embedding)

    ##################################################################################
    # Step 4: Calculate cosine similarity between the sentence and each definition
    ##################################################################################

    similarities = []
    for definition_embedding in definition_embeddings:
        similarity = cosine_similarity(sentence_embedding.reshape(1, -1), definition_embedding.reshape(1, -1))
        similarities.append(similarity)

    ##################################################################################
    # Step 5: Find the most similar definition and return the synset
    ##################################################################################
    
    most_similar_index = similarities.index(max(similarities))

    return synsets[most_similar_index]

def get_GlossBert(sentence, target_word, defition):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelGlossBert.to(device)
    modelGlossBert.eval()

    sentence = sentence.replace(target_word, f"\"{target_word}\"")

    # Tokenize the input sentence and definition
    input = tokenizerGlossBert(
        sentence,
        defition,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    input = {key: value.to(device) for key, value in input.items()}
    with torch.no_grad():
        outputs = modelGlossBert(**input)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return probabilities

def word_sense_disambiguation_GlossBERT(sentence, target_word, lemma, aug=False, pos_tag=None):

    # Get the list of synsets for the target word
    synsets = get_list_of_synsets(lemma, get_wordnet_pos(pos_tag))
    if len(synsets) == 0:
        print(sentence)
        print(f"No synset found for '{lemma}' '{target_word}' '{pos_tag}'.")
        return None

    predictions = []
    for synset in synsets:
        if aug:
            examples = " For Example: " + "; ".join(synset.examples()) if synset.examples() else ""
        else:
            examples = ""
        definition = synset.definition() + examples
        prediction = get_GlossBert(sentence, target_word, definition)
        predictions.append(prediction)

    maximum = 0
    predicted_index = 0
    for i, prediction in enumerate(predictions):
        if prediction[1] > maximum:
            maximum = prediction[1]
            predicted_index = i

    return synsets[predicted_index]
