import os
import json
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

START ="<start>"
END ="<end>"

_non_alnum = re.compile(r"[^a-z0-9?'.,! ]+")
_multi_space = re.compile(r"\s+")

def clean(s: str) -> str:
 s = str(s).lower().strip()
 s = _non_alnum.sub(" ", s)
 s = _multi_space.sub(" ", s).strip()
 return s

def load_artifacts(model_dir: str):
 with open(os.path.join(model_dir, r'C:\Users\Sakthi\PycharmProjects\chatbot\src\artifacts\tokenizer.json'), 'r', encoding='utf-8')as f:
  tok = tokenizer_from_json(f.read())
 with open(os.path.join(model_dir, r'C:\Users\Sakthi\PycharmProjects\chatbot\src\artifacts\meta.json'), 'r') as f:
  meta = json.load(f)
 model = tf.keras.models.load_model(r"C:\Users\Sakthi\PycharmProjects\chatbot\src\artifacts\seq2seq.keras")
 return model, tok, meta

def ids_to_word(tokenizer):
 return {idx: w for w, idx in tokenizer.word_index.items()}

def greedy_decode(model, tokenizer, meta, prompt: str,max_new_tokens: int = 30):
 max_len = int(meta['max_len'])
 w2i = tokenizer.word_index
 i2w = ids_to_word(tokenizer)
 enc = tokenizer.texts_to_sequences([clean(prompt)])
 enc = pad_sequences(enc, maxlen=max_len, padding='post')
 # decoder starts with <start>
 dec_tokens = [w2i.get(START, 0)]
 for _ in range(max_new_tokens):
    dec_in = pad_sequences([dec_tokens], maxlen=max_len,padding='post')
    preds = model.predict([enc, dec_in], verbose=0)
    next_id = int(np.argmax(preds[0, len(dec_tokens)-1])) # last timestep
    if i2w.get(next_id, '') == END:
        break
    dec_tokens.append(next_id)
 words = [i2w.get(i, '') for i in dec_tokens if i not in (0, w2i.get(START, -1))]
 return " ".join([w for w in words if w]).strip()