# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 19:51:09 2025

@author: Adilet
"""

# Imports
import json, re, random
from pathlib import Path
from nltk import pos_tag
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import sequence_accuracy_score

# ----------------------
# (Step 1) - Load JSON
# ----------------------
DATA_PATH = Path(__file__).parent / "adilet.json"
print("(Step 1) - Loading data from:", DATA_PATH)

raw = json.loads(DATA_PATH.read_text(encoding="utf-8"))
# The JSON is a dict with one top-level intent & value is a list of items with data
intent = list(raw.keys())[0]
items = raw[intent]
print("Intent:", intent, " | #instances:", len(items))

# ----------------------
# (Step 2) - Build tokens + BIO labels per sentence
# ----------------------
token_pat = re.compile(r"\w+|[^\w\s]")  # words or single punctuation

def tokenize(txt):
    return token_pat.findall(txt)

def to_bio_sequence(chunks):
    tokens = []
    labels = []
    for ch in chunks:
        txt = ch.get("text", "")
        ent = ch.get("entity", None)
        toks = tokenize(txt)
        if ent:
            tag_b = "B-" + ent
            tag_i = "I-" + ent
            for i, tk in enumerate(toks):
                labels.append(tag_b if i == 0 else tag_i)
                tokens.append(tk)
        else:
            for tk in toks:
                labels.append("O")
                tokens.append(tk)
    return tokens, labels

X_tokens = []
Y_labels = []
for it in items:
    toks, labs = to_bio_sequence(it["data"])
    if len(toks) == 0:
        continue
    X_tokens.append(toks)
    Y_labels.append(labs)

print("(Step 2) - Built sequences. #sentences:", len(X_tokens))

# ----------------------
# (Step 3) - POS tagging
# ----------------------
def pos_tag_sentence(tokens):
    # nltk.pos_tag expects text tokens
    return [p for (_, p) in pos_tag(tokens)]

X_pos = [pos_tag_sentence(toks) for toks in X_tokens]

# ----------------------
# (Step 4) - Feature extraction per token
# ----------------------
def word2features(tokens, pos_tags, i):
    w = tokens[i]
    p = pos_tags[i]
    feats = {
        "word": w,
        "pos": p,
        "is_first": i == 0,
        "has_digit": any(ch.isdigit() for ch in w),
    }
    if i > 0:
        feats["prev_word"] = tokens[i-1]
        feats["prev_pos"]  = pos_tags[i-1]
    else:
        feats["prev_word"] = "__BOS__"
        feats["prev_pos"]  = "__BOS__"
    if i < len(tokens)-1:
        feats["next_word"] = tokens[i+1]
        feats["next_pos"]  = pos_tags[i+1]
    else:
        feats["next_word"] = "__EOS__"
        feats["next_pos"]  = "__EOS__"
    if i+2 < len(tokens):
        feats["next2_word"] = tokens[i+2]
    else:
        feats["next2_word"] = "__EOS2__"
    return feats

def sent2features(tokens, pos_tags):
    return [word2features(tokens, pos_tags, i) for i in range(len(tokens))]

X_feats = [sent2features(toks, pos) for toks, pos in zip(X_tokens, X_pos)]

# ----------------------
# (Step 5) - train/test split (82/18)
# ----------------------
SEED = 91
rand = random.Random(SEED)
indices = list(range(len(X_feats)))
rand.shuffle(indices)

split_idx = int(0.82 * len(indices))
train_idx = indices[:split_idx]
test_idx  = indices[split_idx:]

X_train = [X_feats[i] for i in train_idx]
Y_train = [Y_labels[i] for i in train_idx]
X_test  = [X_feats[i] for i in test_idx]
Y_test  = [Y_labels[i] for i in test_idx]

print("(Step 5) - Split 82/18")
print("Training instances:", len(X_train))
print("Testing instances:", len(X_test))

# ----------------------
# (Step 6) - Train CRF
# ----------------------
print("(Step 6) - Training CRF (lbfgs, c1=0.0025, c2=3, max_iterations=80)")
crf_adilet = CRF(
    algorithm='lbfgs',
    c1=0.0025,
    c2=3,
    max_iterations=80,
    all_possible_transitions=True
)
crf_adilet.fit(X_train, Y_train)

# ----------------------
# (Step 7) - Evaluate
# ----------------------
print("(Step 7) - Evaluating on test set")
Y_pred = crf_adilet.predict(X_test)
acc = sequence_accuracy_score(Y_test, Y_pred)
print("Sequence Accuracy:", acc)

# Retrieve 18th prediction from testing (1-based indexing from spec -> 18th)
if len(X_test) >= 18:
    idx = 17  # 0-based
    print("\n(18th test instance) Tokens:")
    print(" ", [t["word"] for t in X_test[idx]])
    print("(18th) True labels:")
    print(" ", Y_test[idx])
    print("(18th) Predicted labels:")
    print(" ", Y_pred[idx])
else:
    print("\nNote: Test set has fewer than 18 instances.")

# ----------------------
# (Step 8) - new instance prediction
# ----------------------
new_tokens = ['I ', 'want', 'to', 'book', 'for', 'five', 'people', 'a', 'table', 'at', 'Le Cinq restaurant', 'in', 'Paris']
# Normalize to tokenization style used above (split the 'Le Cinq restaurant' into tokens)
def normalize_list_tokens(lst):
    out = []
    for t in lst:
        out.extend(re.findall(r"\w+|[^\w\s]", t))
    return out

new_tok = normalize_list_tokens(new_tokens)
new_pos = pos_tag(new_tok)
new_pos = [p for (_, p) in new_pos]
new_feats = sent2features(new_tok, new_pos)

new_pred = crf_adilet.predict_single(new_feats)

print("\n(New instance) Tokens:")
print(" ", new_tok)
print("(New instance) Prediction:")
print(" ", new_pred)
