import re
import emoji

def preprocess(text: str) -> str:
    text = str(text)
    text = text.replace("’", "'").replace("‘", "'")
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9'_\s!?]", " ", text)
    return re.sub(r"\s+", " ", text).strip()