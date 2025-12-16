import torch
import torch.nn as nn
import math

# Paramètres du modèle
vocab_size = 10000
embed_dim = 512  # Ajusté pour être divisible par num_heads (512 % 8 = 0)
num_heads = 8
ff_dim = 2048
num_layers = 4
max_length = 256
num_classes = 6
dropout = 0.1

# Note: embed_dim doit être divisible par num_heads
# Si vous changez ces valeurs, assurez-vous que embed_dim % num_heads == 0
# Exemples de combinaisons valides:
#   embed_dim=128, num_heads=8 (128/8=16)
#   embed_dim=256, num_heads=8 (256/8=32)
#   embed_dim=512, num_heads=8 (512/8=64)
#   embed_dim=500, num_heads=4 (500/4=125) ou num_heads=5 (500/5=100)


class PositionalEncoding(nn.Module):
    """Encodage positionnel sinusoïdal pour les séquences."""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Créer une matrice de position (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Appliquer sin aux positions paires
        pe[:, 0::2] = torch.sin(position * div_term)
        # Appliquer cos aux positions impaires
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Ajouter une dimension pour le batch et enregistrer comme buffer
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor de shape (batch_size, seq_len, d_model)
        Returns:
            Tensor avec encodage positionnel ajouté
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class MultiHeadAttention(nn.Module):
    """Mécanisme d'attention multi-têtes."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            # Ajuster num_heads pour qu'il soit compatible avec d_model
            # Trouver le plus grand diviseur de d_model qui est <= num_heads
            best_num_heads = num_heads
            for i in range(num_heads, 0, -1):
                if d_model % i == 0:
                    best_num_heads = i
                    break
            
            if best_num_heads != num_heads:
                import warnings
                warnings.warn(
                    f"d_model ({d_model}) n'est pas divisible par num_heads ({num_heads}). "
                    f"Ajustement automatique de num_heads à {best_num_heads}.",
                    UserWarning
                )
                num_heads = best_num_heads
            else:
                # Si aucun diviseur trouvé, ajuster d_model
                adjusted_d_model = ((d_model + num_heads - 1) // num_heads) * num_heads
                raise ValueError(
                    f"d_model ({d_model}) doit être divisible par num_heads ({num_heads}). "
                    f"Suggestion: utilisez d_model={adjusted_d_model} ou num_heads={d_model // (d_model // num_heads)}"
                )
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Projections linéaires pour Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Calcul de l'attention scaled dot-product.
        
        Args:
            Q: Query tensor (batch_size, num_heads, seq_len, d_k)
            K: Key tensor (batch_size, num_heads, seq_len, d_k)
            V: Value tensor (batch_size, num_heads, seq_len, d_k)
            mask: Masque d'attention optionnel de shape (batch_size, seq_len) ou (batch_size, 1, 1, seq_len)
        Returns:
            Output tensor et attention weights
        """
        # Calculer les scores d'attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Appliquer le masque si fourni
        if mask is not None:
            # Si le masque est 2D (batch_size, seq_len), le transformer en 4D pour le broadcasting
            if mask.dim() == 2:
                # mask: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
                # On masque les positions où mask == 0 (padding)
                mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            
            # Masquer les scores : mettre -inf où mask == 0
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Appliquer les poids d'attention aux valeurs
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: Tensors de shape (batch_size, seq_len, d_model)
            mask: Masque d'attention optionnel
        Returns:
            Output tensor de shape (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)
        
        # Projections linéaires et reshape pour multi-têtes
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Calculer l'attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concaténer les têtes
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Projection finale
        output = self.W_o(attention_output)
        
        return output


class FeedForward(nn.Module):
    """Réseau feed-forward avec activation GELU."""
    
    def __init__(self, d_model, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        """
        Args:
            x: Tensor de shape (batch_size, seq_len, d_model)
        Returns:
            Tensor de shape (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderBlock(nn.Module):
    """Bloc encodeur combinant attention multi-têtes et feed-forward."""
    
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor de shape (batch_size, seq_len, d_model)
            mask: Masque d'attention optionnel
        Returns:
            Tensor de shape (batch_size, seq_len, d_model)
        """
        # Attention avec résidus et normalisation
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward avec résidus et normalisation
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class TransformerModel(nn.Module):
    """Modèle Transformer complet pour la classification d'émotions."""
    
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, 
                 max_length, num_classes, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Embeddings de tokens
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Encodage positionnel
        self.positional_encoding = PositionalEncoding(embed_dim, max_length)
        
        # Blocs d'encodeur
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Dropout global
        self.dropout = nn.Dropout(dropout)
        
        # Couche de classification
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor de tokens de shape (batch_size, seq_len)
            mask: Masque d'attention optionnel
        Returns:
            Logits de shape (batch_size, num_classes)
        """
        # Embeddings de tokens
        x = self.token_embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # Encodage positionnel
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Passer à travers les blocs d'encodeur
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask)
        
        # Pooling : moyenne des embeddings sur la dimension de séquence
        # On peut aussi utiliser x[:, 0, :] pour utiliser le premier token
        x = x.mean(dim=1)  # (batch_size, embed_dim)
        
        # Classification
        logits = self.classifier(x)  # (batch_size, num_classes)
        
        return logits
    
    def predict_proba(self, x, mask=None):
        """
        Retourne les probabilités après softmax.
        
        Args:
            x: Tensor de tokens de shape (batch_size, seq_len)
            mask: Masque d'attention optionnel
        Returns:
            Probabilités de shape (batch_size, num_classes)
        """
        logits = self.forward(x, mask)
        return torch.softmax(logits, dim=-1)
