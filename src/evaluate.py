import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from src.model import TransformerModel, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_length, num_classes, dropout
from src.data import dataset, create_dataloaders, clean_text
from src.tokenizer import SimpleTokenizer

# Mapping des labels
LABEL_NAMES = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}


def load_model_and_tokenizer(checkpoint_path='checkpoints/best_model.pt', tokenizer_path='checkpoints/tokenizer.pt'):
    """Charge le modèle et le tokenizer depuis les checkpoints."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Charger le checkpoint du modèle
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Charger le tokenizer
    tokenizer_data = torch.load(tokenizer_path, map_location=device)
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    tokenizer.word_to_idx = tokenizer_data['word_to_idx']
    tokenizer.idx_to_word = tokenizer_data['idx_to_word']
    
    # Créer le modèle
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
    
    # Charger les poids du modèle
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Modèle chargé depuis {checkpoint_path}")
    print(f"Époque: {checkpoint.get('epoch', 'N/A')}")
    print(f"Validation Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"Validation Accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    return model, tokenizer, device


def evaluate_model(model, val_loader, device):
    """Évalue le modèle sur l'ensemble de validation."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for tokens, labels, mask in tqdm(val_loader, desc="Évaluation"):
            tokens = tokens.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            
            # Forward pass
            logits = model(tokens, mask)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)


def plot_precision_by_label(y_true, y_pred, save_dir='results'):
    """Génère un graphique en barres de la précision par label."""
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    labels = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, precision, color='steelblue', alpha=0.7)
    plt.xlabel('Émotion', fontsize=12)
    plt.ylabel('Précision', fontsize=12)
    plt.title('Précision par Label sur l\'Ensemble de Validation', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for i, (bar, prec) in enumerate(zip(bars, precision)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prec:.3f}\n(n={support[i]})',
                ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/precision_by_label.png', dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {save_dir}/precision_by_label.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_dir='results'):
    """Génère une heatmap de la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    labels = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Nombre d\'échantillons'})
    
    plt.xlabel('Prédiction', fontsize=12)
    plt.ylabel('Vraie Label', fontsize=12)
    plt.title('Matrice de Confusion - Ensemble de Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Matrice de confusion sauvegardée: {save_dir}/confusion_matrix.png")
    plt.close()


def plot_metrics_by_label(y_true, y_pred, save_dir='results'):
    """Génère un graphique comparant precision, recall et F1-score par label."""
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    labels = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Précision', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Rappel', color='coral', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='mediumseagreen', alpha=0.8)
    
    ax.set_xlabel('Émotion', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Métriques par Label (Précision, Rappel, F1-Score)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/metrics_by_label.png', dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {save_dir}/metrics_by_label.png")
    plt.close()


def main():
    print("="*60)
    print("ÉVALUATION DU MODÈLE TRANSFORMER")
    print("="*60)
    
    # Charger le modèle et le tokenizer
    model, tokenizer, device = load_model_and_tokenizer()
    
    # Créer les DataLoaders
    print("\nCréation des DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, tokenizer, batch_size=64, max_length=max_length
    )
    
    # Évaluer le modèle
    print("\nÉvaluation sur l'ensemble de validation...")
    y_pred, y_true = evaluate_model(model, val_loader, device)
    
    # Calculer l'accuracy globale
    accuracy = (y_pred == y_true).mean() * 100
    print(f"\n{'='*60}")
    print(f"ACCURACY GLOBALE: {accuracy:.2f}%")
    print(f"{'='*60}")
    
    # Rapport de classification détaillé
    print("\n" + "="*60)
    print("RAPPORT DE CLASSIFICATION DÉTAILLÉ")
    print("="*60)
    target_names = [LABEL_NAMES[i] for i in range(num_classes)]
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print(report)
    
    # Calculer les métriques par label
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    print("\n" + "="*60)
    print("MÉTRIQUES PAR LABEL")
    print("="*60)
    print(f"{'Émotion':<12} {'Précision':<12} {'Rappel':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    for i in range(len(LABEL_NAMES)):
        print(f"{LABEL_NAMES[i]:<12} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    # Générer les visualisations
    print("\n" + "="*60)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("="*60)
    plot_precision_by_label(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
    plot_metrics_by_label(y_true, y_pred)
    
    print("\n" + "="*60)
    print("ÉVALUATION TERMINÉE")
    print("="*60)
    print(f"Graphiques sauvegardés dans le dossier 'results/'")


if __name__ == "__main__":
    main()

