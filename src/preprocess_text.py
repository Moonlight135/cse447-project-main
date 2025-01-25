import re
import unicodedata


def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Convert to Unicode (preserving all characters in various languages)
    text = unicodedata.normalize('NFKC', text)  # Normalize to Unicode Compatibility
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space

    # Keep only printable characters (you can customize this further)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Cc')  # Remove control characters
    return text.strip()

def preprocess_example(example):
    example['text'] = preprocess_text(example['text'])
    return example

## data now has html tags removed and only kept typable characters
## from pre-processing function



