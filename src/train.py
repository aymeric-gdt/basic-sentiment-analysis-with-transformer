import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from src.model import TransformerModel, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_length, num_classes, dropout
from src.data import dataset, create_dataloaders
from src.tokenizer import SimpleTokenizer


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entraîne le modèle pour une époque."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for tokens, labels, mask in progress_bar:
        tokens = tokens.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(tokens, mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Métriques
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Mettre à jour la barre de progression
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Valide le modèle sur l'ensemble de validation."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        for tokens, labels, mask in progress_bar:
            tokens = tokens.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            
            # Forward pass
            logits = model(tokens, mask)
            loss = criterion(logits, labels)
            
            # Métriques
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Mettre à jour la barre de progression
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main():
    # Configuration
    batch_size = 64
    num_epochs = 20
    learning_rate = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Utilisation du device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Nombre d'époques: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    
    # Construire le tokenizer
    print("\nConstruction du vocabulaire...")
    train_texts = [item['text'] for item in dataset['train']]
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab(train_texts)
    print(f"Vocabulaire construit: {tokenizer.get_vocab_size()} tokens")
    
    # Créer les DataLoaders
    print("\nCréation des DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, tokenizer, batch_size=batch_size, max_length=max_length
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Initialiser le modèle
    print("\nInitialisation du modèle...")
    model = TransformerModel(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        max_length=max_length,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)
    
    # Compter le nombre de paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Nombre total de paramètres: {total_params:,}")
    print(f"Paramètres entraînables: {trainable_params:,}")
    
    # Compiler le modèle : fonction de perte + optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("\nModèle compilé avec:")
    print(f"  - Loss: CrossEntropyLoss")
    print(f"  - Optimizer: Adam (lr={learning_rate})")
    
    # Créer le dossier pour sauvegarder les modèles
    os.makedirs('checkpoints', exist_ok=True)
    
    # Variables pour suivre le meilleur modèle
    best_val_loss = float('inf')
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Boucle d'entraînement
    print("\n" + "="*50)
    print("Début de l'entraînement")
    print("="*50)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nÉpoque {epoch}/{num_epochs}")
        print("-" * 50)
        
        # Entraînement
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Afficher les métriques
        print(f"\nÉpoque {epoch} - Résultats:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Sauvegarder le meilleur modèle (basé sur validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'train_acc': train_acc,
            }
            torch.save(checkpoint, 'checkpoints/best_model.pt')
            
            # Sauvegarder le tokenizer
            tokenizer_data = {
                'word_to_idx': tokenizer.word_to_idx,
                'idx_to_word': tokenizer.idx_to_word,
            }
            torch.save(tokenizer_data, 'checkpoints/tokenizer.pt')
            
            print(f"  ✓ Meilleur modèle sauvegardé! (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "="*50)
    print("Entraînement terminé!")
    print("="*50)
    print(f"\nMeilleures performances:")
    print(f"  Validation Loss: {best_val_loss:.4f}")
    print(f"  Validation Accuracy: {best_val_acc:.2f}%")
    print(f"\nModèle sauvegardé dans: checkpoints/best_model.pt")
    
    # Afficher l'évolution des métriques
    print("\nÉvolution des métriques:")
    print(f"  Train Loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")
    print(f"  Val Loss: {val_losses[0]:.4f} → {val_losses[-1]:.4f}")
    print(f"  Train Acc: {train_accs[0]:.2f}% → {train_accs[-1]:.2f}%")
    print(f"  Val Acc: {val_accs[0]:.2f}% → {val_accs[-1]:.2f}%")


if __name__ == "__main__":
    main()

