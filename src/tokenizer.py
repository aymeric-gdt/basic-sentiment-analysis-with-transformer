from collections import Counter
from typing import List, Dict


class SimpleTokenizer:
    """Tokenizer simple basé sur les mots pour le traitement des textes."""
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        
    def build_vocab(self, texts: List[str]):
        """
        Construit le vocabulaire à partir d'une liste de textes.
        
        Args:
            texts: Liste de textes (chaînes de caractères)
        """
        # Compter les fréquences de tous les mots
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Créer le vocabulaire avec les tokens spéciaux
        self.word_to_idx[self.pad_token] = 0
        self.word_to_idx[self.unk_token] = 1
        
        # Ajouter les mots les plus fréquents
        for word, count in word_counts.most_common(self.vocab_size - 2):
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
        
        # Créer le mapping inverse
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
    def encode(self, text: str, max_length: int = 256) -> List[int]:
        """
        Encode un texte en séquence d'indices.
        
        Args:
            text: Texte à encoder
            max_length: Longueur maximale de la séquence (padding/truncation)
        Returns:
            Liste d'indices
        """
        words = text.lower().split()
        indices = []
        
        for word in words:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx[self.unk_token])
        
        # Truncation
        if len(indices) > max_length:
            indices = indices[:max_length]
        
        # Padding
        while len(indices) < max_length:
            indices.append(self.word_to_idx[self.pad_token])
        
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """
        Décode une séquence d'indices en texte.
        
        Args:
            indices: Liste d'indices
        Returns:
            Texte décodé
        """
        words = []
        for idx in indices:
            if idx in self.idx_to_word:
                word = self.idx_to_word[idx]
                if word not in [self.pad_token, self.unk_token]:
                    words.append(word)
        return " ".join(words)
    
    def get_vocab_size(self) -> int:
        """Retourne la taille du vocabulaire."""
        return len(self.word_to_idx)

