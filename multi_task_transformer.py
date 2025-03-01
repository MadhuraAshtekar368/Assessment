import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

# Task-specific head classes
class ClassificationHead(nn.Module):
    """Head for sentence classification tasks"""
    def __init__(self, input_dim, num_classes, hidden_dim=None, dropout_rate=0.1):
        super(ClassificationHead, self).__init__()
        
        if hidden_dim:
            # Two-layer classification head with non-linearity
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            # Simple linear classifier
            self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, features):
        return self.classifier(features)


class SequenceLabelingHead(nn.Module):
    """Head for token-level prediction tasks like NER"""
    def __init__(self, input_dim, num_labels, use_crf=False):
        super(SequenceLabelingHead, self).__init__()
        self.use_crf = use_crf
        
        # Token-level classifier
        self.token_classifier = nn.Linear(input_dim, num_labels)
        
        # Optional CRF layer could be added here
        if use_crf:
            # Placeholder for CRF - would need a proper CRF implementation
            pass
    
    def forward(self, sequence_output, attention_mask=None):
        # Get token-level predictions
        logits = self.token_classifier(sequence_output)
        
        if self.use_crf and attention_mask is not None:
            # If using CRF, we would apply it here
            # This is just a placeholder
            return logits
        
        return logits


class SentimentAnalysisHead(nn.Module):
    """Head specialized for sentiment analysis with fine-grained outputs"""
    def __init__(self, input_dim, num_classes, use_attention=False):
        super(SentimentAnalysisHead, self).__init__()
        self.use_attention = use_attention
        
        if use_attention:
            # Attention mechanism to focus on sentiment-relevant parts
            self.attention = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.Tanh(),
                nn.Linear(input_dim // 2, 1)
            )
        
        # Main sentiment classifier
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, features):
        if self.use_attention:
            # Apply attention mechanism
            attention_weights = F.softmax(self.attention(features), dim=1)
            weighted_features = attention_weights * features
            return self.classifier(weighted_features)
        
        return self.classifier(features)


class MultiTaskSentenceTransformer(nn.Module):
    """
    Enhanced multi-task learning model for NLP tasks 
    with shared backbone and task-specific heads
    """
    def __init__(
        self, 
        model_name='bert-base-uncased', 
        embedding_dim=768,
        task_config={
            'classification': {'num_classes': 3, 'hidden_dim': 256},
            'ner': {'num_labels': 9, 'use_crf': False},
            'sentiment': {'num_classes': 3, 'use_attention': True}
        },
        shared_layer_strategy='hard',  # 'hard' or 'soft'
        return_embeddings=False
    ):
        super(MultiTaskSentenceTransformer, self).__init__()
        
        self.shared_layer_strategy = shared_layer_strategy
        self.return_embeddings = return_embeddings
        
        # Shared transformer backbone
        self.backbone = BertModel.from_pretrained(model_name)
        
        # Shared embedding layer
        self.shared_encoder = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.backbone.config.hidden_size, embedding_dim),
            nn.Tanh()
        )
        
        # For soft parameter sharing, we create task-specific encoders
        if shared_layer_strategy == 'soft':
            self.classification_encoder = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.backbone.config.hidden_size, embedding_dim),
                nn.Tanh()
            )
            
            self.ner_encoder = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.backbone.config.hidden_size, embedding_dim),
                nn.Tanh()
            )
            
            self.sentiment_encoder = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.backbone.config.hidden_size, embedding_dim),
                nn.Tanh()
            )
        
        # Task-specific heads
        self.classification_head = ClassificationHead(
            input_dim=embedding_dim,
            num_classes=task_config['classification']['num_classes'],
            hidden_dim=task_config['classification']['hidden_dim']
        )
        
        self.ner_head = SequenceLabelingHead(
            input_dim=embedding_dim,
            num_labels=task_config['ner']['num_labels'],
            use_crf=task_config['ner']['use_crf']
        )
        
        self.sentiment_head = SentimentAnalysisHead(
            input_dim=embedding_dim,
            num_classes=task_config['sentiment']['num_classes'],
            use_attention=task_config['sentiment']['use_attention']
        )
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """Calculate mean pooling with attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_ids, attention_mask, task=None):
        """
        Forward pass with optional task-specific processing
        
        Args:
            input_ids: Tensor of token ids
            attention_mask: Tensor of attention masks
            task: Optional task name to return only specific outputs
                  (classification, ner, sentiment, or all)
        
        Returns:
            Dictionary of task outputs or specific task output based on 'task' parameter
        """
        # Get contextualized token representations from BERT
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Hard parameter sharing: use the same encoder for all tasks
        if self.shared_layer_strategy == 'hard':
            # For sentence-level tasks, pool the token outputs
            pooled_output = self._mean_pooling(sequence_output, attention_mask)
            
            # Apply shared encoder to get embeddings
            shared_embedding = self.shared_encoder(pooled_output)
            shared_embedding = F.normalize(shared_embedding, p=2, dim=1)
            
            # Get normalized token embeddings for sequence tasks
            token_embeddings = self.shared_encoder(sequence_output)
            
            # Run task-specific heads
            classification_output = self.classification_head(shared_embedding)
            ner_output = self.ner_head(token_embeddings, attention_mask)
            sentiment_output = self.sentiment_head(shared_embedding)
        
        # Soft parameter sharing: use task-specific encoders
        elif self.shared_layer_strategy == 'soft':
            # For sentence-level tasks, pool the token outputs
            pooled_output = self._mean_pooling(sequence_output, attention_mask)
            
            # Apply task-specific encoders
            classification_embedding = F.normalize(self.classification_encoder(pooled_output), p=2, dim=1)
            ner_embeddings = self.ner_encoder(sequence_output)
            sentiment_embedding = F.normalize(self.sentiment_encoder(pooled_output), p=2, dim=1)
            
            # Run task-specific heads
            classification_output = self.classification_head(classification_embedding)
            ner_output = self.ner_head(ner_embeddings, attention_mask)
            sentiment_output = self.sentiment_head(sentiment_embedding)
        
        # Return specific task output or all outputs
        if task == 'classification':
            return classification_output
        elif task == 'ner':
            return ner_output
        elif task == 'sentiment':
            return sentiment_output
        else:
            # Return all outputs and optionally embeddings
            results = {
                'classification': classification_output,
                'ner': ner_output,
                'sentiment': sentiment_output
            }
            
            if self.return_embeddings:
                if self.shared_layer_strategy == 'hard':
                    results['embeddings'] = shared_embedding
                else:
                    results['classification_embedding'] = classification_embedding
                    results['sentiment_embedding'] = sentiment_embedding
            
            return results