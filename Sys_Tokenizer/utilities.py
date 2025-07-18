import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# ============================================
# ✅ Accurate Burmese Syllable Tokenization
# ============================================
def syllable_tokenization(text: str) -> str:
    """
    Tokenize Burmese text into syllables using the regex from the Perl syllbreak-unicode.pl script.
    """
    pattern = re.compile(
        r"(([A-Za-z0-9]+)|[က-အ|ဥ|ဦ](င်္|[က-အ][့း]*[်]|္[က-အ]|[ါ-ှႏꩻ][ꩻ]*){0,}|.)",
        re.UNICODE
    )
    tokenized = pattern.sub(r"\1 ", text.strip())
    return tokenized.strip()

# ============================================
# 🔤 Character-level Tokenization
# ============================================
def character_tokenization(text: str) -> str:
    return re.sub(r"([^\s])", r"\1      ", text)

# ============================================
# 🌐 Multilingual Semi-Syllable Break
# ============================================
def multilingual_semi_syllable_break(text: str) -> str:
    result = re.sub(r"([a-zA-Z]+|[຀-႞ꩻ][ະ-ꩻ]{0,}|.)", r"\1.....", text)
    result = re.sub(r" +", " ", result)
    return result

# ============================================
# 🧠 Zawgyi-Unicode Detection (TensorFlow)
# ============================================
def load_zawgyi_unicode_detection_model():
    model_path = "model/zawgyi-unicode-detection/zawgyiunicodedetectionstreamlit.h5"
    return tf.keras.models.load_model(model_path)

def load_zawgyi_unicode_tokenizer():
    with open('model/zawgyi-unicode-detection/tokenizer.pickle', 'rb') as file:
        return pickle.load(file)

def zawgyi_unicode_detection(text: str) -> str:
    tokenizer = load_zawgyi_unicode_tokenizer()
    model = load_zawgyi_unicode_detection_model()
    seqs = tokenizer.texts_to_sequences([syllable_tokenization(text)])
    padded = pad_sequences(seqs, maxlen=150, truncating='post', padding='post')
    prediction = model.predict(padded)[0][0]
    return "Unicode Encoding" if prediction >= 0.5 else "Zawgyi Encoding"

# ============================================
# 🔍 Keywords Detection
# ============================================
def keywords_detection(lexicon: str, text: str):
    keywords = ""
    for i in lexicon.strip().lower().split("|||"):
        keywords += i.strip().lower().replace("$", r"\$").replace(" ", "_") + r"(?![ါ-ှ]|[က-အ]်)" + "|"
    keywords = keywords.rstrip("|")
    text = re.sub(r" ", r"_", text.strip())
    return re.findall(f"{keywords}", text.lower())

# ============================================
# 🔁 N-Grams Generator
# ============================================
def n_grams(k: int, text: str, option: str):
    if k < 1:
        return ""

    if option == "Character":
        i = character_tokenization(text)
    else:
        i = syllable_tokenization(text)

    i = i.strip().split(" ")

    if k > len(i):
        k = len(i)

    prev = i[0:k]
    result = ''.join(prev)

    for j in range(k, len(i)):
        prev = prev[1:] + [i[j]]
        result += "--------" + ''.join(prev)

    return result

# ============================================
# 🧽 Remove Characters from Text
# ============================================
def remove_chars(chars: str, text: str) -> str:
    return ''.join([ch for ch in text if ch not in chars])

# ============================================
# ✅ Validate Balanced Parentheses
# ============================================
def valid_parentheses(text: str) -> bool:
    open_brackets = {"{", "(", "["}
    matching = {"}": "{", ")": "(", "]": "["}
    stack = []

    for ch in text:
        if ch in open_brackets:
            stack.append(ch)
        elif ch in matching:
            if not stack or stack.pop() != matching[ch]:
                return False

    return not stack
