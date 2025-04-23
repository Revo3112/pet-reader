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
    "pit_bull": "Dog",  # Tambahkan ini khusus untuk pit bull terrier
    "terrier": "Dog"    # Tambahkan ini sebagai fallback untuk semua terrier
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
    
    # Coba cari kata kunci dari nama breed dalam dictionary
    for key in pet_mapping:
        # Cek apakah key merupakan bagian dari nama breed
        if key.lower() in breed_lower:
            return pet_mapping[key]
    
    # Jika tidak ada yang cocok, kembalikan default (anggap anjing)
    print(f"Warning: Mapping not found for breed: {breed}, defaulting to Dog")
    return "Dog"