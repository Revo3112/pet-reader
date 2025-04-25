pet_mapping = {
    "Abyssinian": "Cat",
    "american": "Dog",
    "basset": "Dog",
    "beagle": "Dog",
    "Bengal": "Cat",
    "Birman": "Cat",
    "Bombay": "Cat",
    "boxer": "Dog",
    "British": "Cat",
    "chihuahua": "Dog",
    "Egyptian": "Cat",
    "english": "Dog",
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
    "yorkshire": "Dog",
    "pit_bull": "Dog",  # Untuk pit bull terrier
    "terrier": "Dog",   # Fallback untuk semua terrier
    "coon": "Cat",      # Untuk Maine Coon
    "shorthair": "Cat", # Untuk British Shorthair
    "blue": "Cat",      # Untuk Russian Blue
    "doll": "Cat"       # Untuk Ragdoll (tambahan)
}

def get_mapping(breed):
    """Mendapatkan jenis hewan (Dog/Cat) berdasarkan nama ras."""
    # Coba langsung dengan kunci yang ada
    if breed in pet_mapping:
        return pet_mapping[breed]
    
    # Jika tidak ditemukan, cek apakah ada bagian dari nama ras dalam kamus
    breed_lower = breed.lower()
    
    # Hilangkan awalan seperti "purebred" jika ada
    if "purebred" in breed_lower:
        breed_lower = breed_lower.replace("purebred", "").strip()
    
    # Ganti underscore dengan spasi
    breed_lower = breed_lower.replace("_", " ")
    
    # Coba cari kata kunci dari nama breed dalam dictionary
    for key in pet_mapping:
        # Cek apakah key merupakan bagian dari nama breed
        if key.lower() in breed_lower:
            return pet_mapping[key]
    
    # Jika tidak ada yang cocok, coba analisis nama
    if any(cat_term in breed_lower for cat_term in ["cat", "kitty", "kitten", "feline"]):
        return "Cat"
    if any(dog_term in breed_lower for dog_term in ["dog", "puppy", "canine", "hound"]):
        return "Dog"
    
    # Default fallback
    print(f"Warning: Mapping not found for breed: {breed}, defaulting to Dog")
    return "Dog"