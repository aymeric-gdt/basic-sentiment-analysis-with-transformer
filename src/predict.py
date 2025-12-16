import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.model import TransformerModel, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_length, num_classes, dropout
from src.data import clean_text
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
    
    return model, tokenizer, device


def predict_emotion(text, model, tokenizer, device, show_probs=True):
    """
    Prédit l'émotion d'un texte.
    
    Args:
        text: Texte à analyser
        model: Modèle Transformer entraîné
        tokenizer: Tokenizer
        device: Device (CPU ou GPU)
        show_probs: Afficher les probabilités pour toutes les classes
    Returns:
        Tuple (label_prédit, nom_émotion, probabilités)
    """
    # Nettoyer le texte
    cleaned_text = clean_text(text)
    
    # Tokeniser
    tokens = tokenizer.encode(cleaned_text, max_length=max_length)
    tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    
    # Créer le masque
    mask = (tokens_tensor != 0).long().to(device)
    
    # Prédiction
    model.eval()
    with torch.no_grad():
        logits = model(tokens_tensor, mask)
        probs = torch.softmax(logits, dim=-1)
        predicted_label = torch.argmax(probs, dim=1).item()
    
    emotion_name = LABEL_NAMES[predicted_label]
    probabilities = probs[0].cpu().numpy()
    
    return predicted_label, emotion_name, probabilities


def display_prediction(text, predicted_label, emotion_name, probabilities):
    """Affiche les résultats de la prédiction."""
    print("\n" + "="*60)
    print("RÉSULTAT DE LA PRÉDICTION")
    print("="*60)
    print(f"Texte analysé: {text}")
    print(f"\nÉmotion prédite: {emotion_name.upper()} (label: {predicted_label})")
    print(f"\nProbabilités pour toutes les classes:")
    print("-" * 60)
    for i, (label, name) in enumerate(LABEL_NAMES.items()):
        prob = probabilities[i] * 100
        marker = " <-- PRÉDIT" if i == predicted_label else ""
        print(f"  {name:12s}: {prob:6.2f}%{marker}")
    print("="*60)


def plot_probabilities(probabilities, save_path=None):
    """Génère un graphique en barres des probabilités."""
    labels = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, probabilities * 100, color='steelblue', alpha=0.7)
    
    # Mettre en évidence la classe prédite
    predicted_idx = np.argmax(probabilities)
    bars[predicted_idx].set_color('coral')
    bars[predicted_idx].set_alpha(0.9)
    
    plt.xlabel('Émotion', fontsize=12)
    plt.ylabel('Probabilité (%)', fontsize=12)
    plt.title('Probabilités de Prédiction par Émotion', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{prob*100:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold' if i == predicted_idx else 'normal')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nGraphique sauvegardé: {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    """Fonction principale pour tester le modèle."""
    import sys
    
    print("="*60)
    print("PRÉDICTION D'ÉMOTION - MODÈLE TRANSFORMER")
    print("="*60)
    
    # Charger le modèle et le tokenizer
    print("\nChargement du modèle et du tokenizer...")
    try:
        model, tokenizer, device = load_model_and_tokenizer()
        print("✓ Modèle et tokenizer chargés avec succès!")
    except FileNotFoundError as e:
        print(f"❌ Erreur: {e}")
        print("Assurez-vous d'avoir entraîné le modèle et que les fichiers existent:")
        print("  - checkpoints/best_model.pt")
        print("  - checkpoints/tokenizer.pt")
        return
    
    # Mode interactif ou argument en ligne de commande
    if len(sys.argv) > 1:
        # Texte fourni en argument
        text = " ".join(sys.argv[1:])
    else:
        # Mode interactif
        print("\nMode interactif - Entrez 'quit' pour quitter")
        print("-" * 60)
        text = input("\nEntrez un texte à analyser: ").strip()
        if not text or text.lower() == 'quit':
            print("Au revoir!")
            return
    
    # Prédiction
    predicted_label, emotion_name, probabilities = predict_emotion(text, model, tokenizer, device)
    
    # Afficher les résultats
    display_prediction(text, predicted_label, emotion_name, probabilities)
    
    # Générer le graphique
    plot_probabilities(probabilities)


if __name__ == "__main__":
    main()

