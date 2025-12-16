from datasets import load_dataset
import re
from unidecode import unidecode
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from src.tokenizer import SimpleTokenizer

dataset = load_dataset("dair-ai/emotion")

print(dataset) # --> DatasetDict

#DatasetDict({
#    train: Dataset({
#        features: ['text', 'label'],
#        num_rows: 16000
#    })
#    validation: Dataset({
#        features: ['text', 'label'],
#        num_rows: 2000
#    })
#    test: Dataset({
#        features: ['text', 'label'],
#        num_rows: 2000
#    })
#})

# Nettoyage des données : 
# normalisation des accents, suppression des URL, des mentions, des hashtags et des caractères spéciaux

def clean_text(text):
    # suppression des URL
    text = re.sub(r'https?://\S+', '', text)
    # normalisation des accents
    text = unidecode(text)
    # suppression des mentions
    text = re.sub(r'@\S+', '', text)
    # suppression des hashtags
    text = re.sub(r'#\S+', '', text)
    # suppression des caractères spéciaux
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

dataset['train'] = dataset['train'].map(lambda x: {'text': clean_text(x['text'])})
dataset['validation'] = dataset['validation'].map(lambda x: {'text': clean_text(x['text'])})
dataset['test'] = dataset['test'].map(lambda x: {'text': clean_text(x['text'])})

print(dataset['train'][1])

print(dataset['validation'][1])

print(dataset['test'][1])

# verification des labels : 
# 0 -> sadness
# 1 -> joy
# 2 -> love
# 3 -> anger
# 4 -> fear
# 5 -> surprise


class EmotionDataset(Dataset):
    """Dataset PyTorch pour les données d'émotions."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: SimpleTokenizer, max_length: int = 256):
        """
        Args:
            texts: Liste de textes
            labels: Liste de labels (entiers)
            tokenizer: Tokenizer pour encoder les textes
            max_length: Longueur maximale des séquences
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encoder le texte
        tokens = self.tokenizer.encode(text, self.max_length)
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """
    Fonction de collation pour créer les batches avec masques d'attention.
    
    Args:
        batch: Liste de tuples (tokens, label)
    Returns:
        tokens: Tensor de shape (batch_size, seq_len)
        labels: Tensor de shape (batch_size,)
        mask: Tensor de masque d'attention (batch_size, seq_len)
    """
    tokens, labels = zip(*batch)
    tokens = torch.stack(tokens)
    labels = torch.stack(labels)
    
    # Créer le masque d'attention (1 pour les tokens réels, 0 pour le padding)
    # Le token PAD a l'indice 0
    mask = (tokens != 0).long()
    
    return tokens, labels, mask


def create_dataloaders(dataset_dict, tokenizer: SimpleTokenizer, batch_size: int = 64, max_length: int = 256):
    """
    Crée les DataLoaders pour l'entraînement, la validation et le test.
    
    Args:
        dataset_dict: DatasetDict de HuggingFace datasets
        tokenizer: Tokenizer initialisé et avec vocabulaire construit
        batch_size: Taille des batches
        max_length: Longueur maximale des séquences
    Returns:
        Tuple de (train_loader, val_loader, test_loader)
    """
    # Extraire les textes et labels
    train_texts = [item['text'] for item in dataset_dict['train']]
    train_labels = [item['label'] for item in dataset_dict['train']]
    
    val_texts = [item['text'] for item in dataset_dict['validation']]
    val_labels = [item['label'] for item in dataset_dict['validation']]
    
    test_texts = [item['text'] for item in dataset_dict['test']]
    test_labels = [item['label'] for item in dataset_dict['test']]
    
    # Créer les datasets
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, max_length)
    
    # Créer les DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader