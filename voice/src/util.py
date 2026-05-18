import unicodedata


def normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = unicodedata.normalize("NFC", name)
    return name
