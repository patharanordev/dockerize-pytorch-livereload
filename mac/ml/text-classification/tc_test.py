import re
import os
import torch
import torchtext
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import text_classification

import os

DIR_PATH = './ml/text-classification'
DATA_PATH = '{}/.data'.format(DIR_PATH)

if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)

# -------------------------------------------------
# Load data with ngrams

NGRAMS = 2

train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root=DATA_PATH, ngrams=NGRAMS, vocab=None
)
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Custom model : Text sentiment

from text_sentiment_model import TextSentiment

# -------------------------------------------------
# Load model

VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUN_CLASS = len(train_dataset.get_labels())

model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)
# Model class must be defined somewhere (above)
model.load_state_dict(torch.load('{}/tc-model.pt'.format(DIR_PATH)))
model.eval()

# -------------------------------------------------
# Test on a random news

ag_news_label = {1 : "World",
                 2 : "Sports",
                 3 : "Business",
                 4 : "Sci/Tec"}

def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

vocab = train_dataset.get_vocab()
model = model.to("cpu")

print("This is a %s news" %ag_news_label[predict(ex_text_str, model, vocab, 2)])