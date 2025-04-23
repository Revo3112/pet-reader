pet_mapping = {
    "Abyssinian": "Cat",
    "american": "Dog",  # Note: duplicate entry in original data
    "basset": "Dog",
    "beagle": "Dog",
    "Bengal": "Cat",
    "Birman": "Cat",
    "Bombay": "Cat",
    "boxer": "Dog",
    "British": "Cat",
    "chihuahua": "Dog",
    "Egyptian": "Cat",
    "english": "Dog",  # Note: duplicate entry in original data
    "german": "Dog",
    "great": "Dog",
    "havanese": "Dog",
    "japanese": "Dog",
    "keeshond": "Dog",
    "leonberger": "Dog",
    "Maine": "Cat",
    "miniature": "Dog",
    "newfoundland": "Dog",
    "Persian": "Cat",
    "pomeranian": "Dog",
    "pug": "Dog",
    "Ragdoll": "Cat",
    "Russian": "Cat",
    "saint": "Dog",
    "samoyed": "Dog",
    "scottish": "Dog",
    "shiba": "Dog",
    "Siamese": "Cat",
    "staffordshire": "Dog",
    "wheaten": "Dog",
    "yorkshire": "Dog"
}

def get_mapping(breed):
    return pet_mapping[breed]