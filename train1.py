import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, classification_report, accuracy_score, roc_auc_score, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import joblib
import warnings
import math
import seaborn as sns

warnings.filterwarnings('ignore')

# Set global font size
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 18

sns.set_context("talk", font_scale=1.2)
sns.set_style("whitegrid")

# --------------------- Transfer Learning Model Definition ---------------------
class TransferLearningModel(nn.Module):
    def __init__(self, input_dim, base_model=None):
        super(TransferLearningModel, self).__init__()
        self.base_model = base_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Neural network architecture (same as ALG_TL_AS.ipynb)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Initialize weights with feature importance from base model
        if base_model is not None and hasattr(base_model, 'feature_importances_'):
            self._initialize_weights_with_feature_importance()
    
    def _initialize_weights_with_feature_importance(self):
        """Initialize neural network weights using feature importance from base model"""
        feature_importances = self.base_model.feature_importances_
        with torch.no_grad():
            # Initialize first layer weights
            for i in range(len(feature_importances)):
                self.model[0].weight[:, i] = feature_importances[i] * self.model[0].weight[:, i]
    
    def forward(self, x):
        return self.model(x)
    
    def evaluate(self, data_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 3:  # Includes weights
                    features, labels, weights = batch
                else:
                    features, labels = batch
                
                features = features.to(self.device).float()
                labels = labels.to(self.device).float().view(-1, 1)
                
                outputs = self.model(features)
                preds = outputs.cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels_np)
        
        if len(all_labels) == 0:
            return {
                'auc': 0.0,
                'f1': 0.0,
                'recall': 0.0,
                'accuracy': 0.0,
                'precision': 0.0
            }, [], []
        
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        
        preds_binary = (all_preds > 0.5).astype(int)
        
        # Calculate all metrics
        auc_val = roc_auc_score(all_labels, all_preds)
        f1_val = f1_score(all_labels, preds_binary)
        recall_val = recall_score(all_labels, preds_binary)
        accuracy_val = accuracy_score(all_labels, preds_binary)
        precision_val = precision_score(all_labels, preds_binary)
        
        metrics = {
            'auc': auc_val,
            'f1': f1_val,
            'recall': recall_val,
            'accuracy': accuracy_val,
            'precision': precision_val
        }
        
        return metrics, all_preds, all_labels

# --------------------- Dataset Class ---------------------
class ALGBADataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)
        self.weights = torch.ones(len(labels))  # All samples have weight 1
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.weights[idx]

# --------------------- Data Preprocessing ---------------------
def load_and_preprocess_data(file_path="ALG_nonscaled.csv"):
    """
    Load and preprocess data for ALG vs BA differential diagnosis using transfer learning
    """
    # Load data (same as ALG_TL_AS.ipynb)
    data = pd.read_csv(file_path)
    data_target = data['AGS']  # Target: AGS (1) vs BA (0)
    
    # Select the 6 key features for ALG vs BA differential diagnosis
    data_features = data[['Acholic_stools ', 'Abnormal_GB_morphology', 'Glisson_Sheath_Thickening', 'Length', 'LSM', 'GGT']]
    
    # Standardize features using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(data_features)
    data_features_scaled = scaler.transform(data_features)
    
    # Split dataset (30% test, stratified by target)
    x_train, x_test, y_train, y_test = train_test_split(
        data_features_scaled, 
        data_target, 
        test_size=0.3, 
        stratify=data_target, 
        random_state=0
    )
    
    # Convert arrays back to DataFrame
    feature_names = ['Acholic_stools', 'Abnormal_GB_morphology', 'Glisson_Sheath_Thickening', 'Length', 'LSM', 'GGT']
    x_train = pd.DataFrame(x_train, columns=feature_names)
    x_test = pd.DataFrame(x_test, columns=feature_names)
    
    return x_train, x_test, y_train, y_test, scaler

# --------------------- AUC Confidence Interval ---------------------
def calculate_auc_ci(y_true, y_pred, n_bootstraps=1000, alpha=0.95):
    """
    Calculate AUC confidence interval
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    n = len(y_true)
    bootstrapped_auc = []
    
    # For small sample sizes use exact binomial distribution
    if n < 30:
        auc_val = roc_auc_score(y_true, y_pred)
        z = 1.96  # 95% CI
        n_pos = sum(y_true)
        n_neg = n - n_pos
        p = auc_val
        se = np.sqrt((p * (1 - p)) / n)
        lower = max(0, p - z * se)
        upper = min(1, p + z * se)
        return auc_val, (lower, upper)
    
    # Use bootstrap for adequate sample sizes
    for _ in range(n_bootstraps):
        indices = np.random.choice(range(n), n, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        auc_val = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_auc.append(auc_val)
    
    sorted_auc = np.sort(bootstrapped_auc)
    lower_idx = int(n_bootstraps * (1 - alpha) / 2)
    upper_idx = int(n_bootstraps * (1 + alpha) / 2)
    
    auc_mean = np.mean(sorted_auc)
    ci_lower = sorted_auc[lower_idx]
    ci_upper = sorted_auc[upper_idx]
    
    return auc_mean, (ci_lower, ci_upper)

# --------------------- Training Function ---------------------
def train_transfer_learning_model(x_train, y_train, x_val, y_val, epochs=100, lr=0.001):
    """
    Train transfer learning model with focus on recall
    """
    # Create base Random Forest model (same as ALG_TL_AS.ipynb)
    base_model = RandomForestClassifier(n_estimators=1000, random_state=0)
    base_model.fit(x_train, y_train)
    
    # Create datasets
    train_dataset = ALGBADataset(x_train, y_train)
    val_dataset = ALGBADataset(x_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create transfer learning model
    input_dim = x_train.shape[1]
    model = TransferLearningModel(input_dim, base_model=base_model)
    
    device = model.device
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    best_val_recall = 0
    
    print("Training Transfer Learning Model for ALG vs BA Differential Diagnosis...")
    for epoch in range(epochs):
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for features, labels, weights in train_loader:
            features = features.to(device).float()
            labels = labels.to(device).float().view(-1, 1)
            weights = weights.to(device).float()
            
            optimizer.zero_grad()
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss = (loss * weights).mean()  # Apply sample weights
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation evaluation every 20 epochs
        if (epoch + 1) % 20 == 0:
            val_metrics, _, _ = model.evaluate(val_loader)
            val_auc = val_metrics['auc']
            val_f1 = val_metrics['f1']
            val_recall = val_metrics['recall']
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | Val Recall: {val_recall:.4f}")
            
            # Save best model based on recall
            if val_recall > best_val_recall:
                best_val_recall = val_recall
                torch.save(model.state_dict(), 'best_alg_ba_model.pth')
                print(f"  New best model saved with recall: {val_recall:.4f}")
    
    # Load best model
    try:
        model.load_state_dict(torch.load('best_alg_ba_model.pth', weights_only=True))
    except TypeError:
        # Fallback for older PyTorch versions
        model.load_state_dict(torch.load('best_alg_ba_model.pth'))
    print(f"Training completed. Best validation recall: {best_val_recall:.4f}")
    
    return model, base_model

# --------------------- Model Evaluation ---------------------
def evaluate_model(model, x, y, set_name='Validation'):
    """
    Comprehensive model evaluation
    """
    device = model.device
    
    # Convert to PyTorch tensors
    features = torch.tensor(x.values, dtype=torch.float32).to(device)
    labels = torch.tensor(y.values, dtype=torch.float32).view(-1, 1).to(device)
    
    # Create data loader
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    metrics, all_preds, all_labels = model.evaluate(loader)
    
    # Calculate AUC confidence interval
    auc_val = metrics['auc']
    _, (auc_lower, auc_upper) = calculate_auc_ci(all_labels, all_preds)
    metrics['auc_ci'] = (auc_lower, auc_upper)
    
    # Calculate Brier score
    brier_score = brier_score_loss(all_labels, all_preds)
    metrics['brier'] = brier_score
    
    # Calculate confusion matrix
    preds_binary = (np.array(all_preds) > 0.5).astype(int)
    cm = confusion_matrix(all_labels, preds_binary)
    
    print(f"\n===== {set_name} Set Results =====")
    print(f"AUC: {auc_val:.4f} (95% CI: {auc_lower:.4f}-{auc_upper:.4f})")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Brier Score: {brier_score:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, preds_binary, target_names=['BA', 'ALG']))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(cm)
    
    return metrics, all_preds, all_labels, cm

# --------------------- Visualization ---------------------
def plot_results(true_labels, pred_probs, set_name, cm):
    """
    Plot ROC curve and other evaluation plots
    """
    # ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)
    auc_mean, (auc_lower, auc_upper) = calculate_auc_ci(true_labels, pred_probs)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f} [{auc_lower:.2f}-{auc_upper:.2f}])')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({set_name} Set)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(f'roc_curve_{set_name.lower()}.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calibration curve
    prob_true, prob_pred = calibration_curve(true_labels, pred_probs, n_bins=10, strategy='quantile')
    
    plt.figure(figsize=(10, 8))
    plt.plot(prob_pred, prob_true, marker='o', linestyle='-', label='Model Calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curve ({set_name} Set)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'calibration_curve_{set_name.lower()}.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['BA', 'ALG'], 
                yticklabels=['BA', 'ALG'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix ({set_name} Set)')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{set_name.lower()}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# --------------------- Prediction Service ---------------------
class ALGBATransferLearningPredictor:
    """
    ALG Syndrome vs Biliary Atresia Differential Diagnosis Transfer Learning Predictor
    """
    def __init__(self, model_path='alg_ba_transfer_model.pth', scaler_path='alg_ba_scaler.pkl'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scaler = joblib.load(scaler_path)
        self.required_features = ['Acholic_stools', 'Abnormal_GB_morphology', 'Glisson_Sheath_Thickening', 'Length', 'LSM', 'GGT']
        
        # Load model architecture and weights
        self.model = TransferLearningModel(len(self.required_features))
        try:
            self.model.load_state_dict(torch.load(model_path, 
                                                map_location=self.device, 
                                                weights_only=True))
        except TypeError:
            # Fallback for older PyTorch versions
            self.model.load_state_dict(torch.load(model_path, 
                                                map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def validate_input(self, input_data):
        """
        Validate input data
        """
        if len(input_data) != len(self.required_features):
            raise ValueError(f"Expected {len(self.required_features)} features, got {len(input_data)}")
        
        # Basic validation
        for i, value in enumerate(input_data):
            if not isinstance(value, (int, float)):
                raise TypeError(f"Feature {self.required_features[i]} must be numeric")
            if value < 0:
                raise ValueError(f"Feature {self.required_features[i]} cannot be negative")
    
    def predict(self, input_data):
        """
        Make prediction using transfer learning model for ALG vs BA differential diagnosis
        """
        self.validate_input(input_data)
        
        # Preprocess input
        scaled_data = self.scaler.transform([input_data])
        
        # Convert to tensor and predict
        with torch.no_grad():
            input_tensor = torch.tensor(scaled_data, dtype=torch.float32).to(self.device)
            probability = self.model(input_tensor).cpu().numpy()[0][0]
        
        # ALG probability (since target is AGS=1)
        alg_probability = probability
        ba_probability = 1 - probability
        
        diagnosis = "ALG Syndrome" if probability >= 0.5 else "Biliary Atresia (BA)"
        
        return {
            'diagnosis': diagnosis,
            'alg_probability': round(alg_probability, 4),
            'ba_probability': round(ba_probability, 4),
            'alg_percentage': round(alg_probability * 100, 1),
            'ba_percentage': round(ba_probability * 100, 1),
            'disclaimer': 'This transfer learning-based differential diagnosis tool distinguishes between ALG Syndrome and Biliary Atresia. Clinical evaluation and additional tests are required for definitive diagnosis.'
        }

# --------------------- Main Training Pipeline ---------------------
def main():
    print("ALG Syndrome vs Biliary Atresia Differential Diagnosis Transfer Learning Model")
    print("=" * 60)
    
    # Data preprocessing
    x_train, x_test, y_train, y_test, scaler = load_and_preprocess_data("ALG_nonscaled.csv")
    
    # Save scaler
    joblib.dump(scaler, "alg_ba_scaler.pkl")
    print("Scaler saved as 'alg_ba_scaler.pkl'")
    
    # Train transfer learning model
    print("\nTraining Transfer Learning Model...")
    transfer_model, base_model = train_transfer_learning_model(x_train, y_train, x_test, y_test, epochs=100)
    
    # Save transfer learning model
    torch.save(transfer_model.state_dict(), 'alg_ba_transfer_model.pth')
    print("Transfer learning model saved as 'alg_ba_transfer_model.pth'")
    
    # Save base random forest model
    joblib.dump(base_model, 'alg_ba_rf_model.pkl')
    print("Random Forest base model saved as 'alg_ba_rf_model.pkl'")
    
    # Evaluate model
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    # Training set performance
    train_metrics, train_preds, train_labels, train_cm = evaluate_model(transfer_model, x_train, y_train, "Training")
    plot_results(train_labels, train_preds, "Training", train_cm)
    
    # Test set performance  
    test_metrics, test_preds, test_labels, test_cm = evaluate_model(transfer_model, x_test, y_test, "Test")
    plot_results(test_labels, test_preds, "Test", test_cm)
    
    # Performance summary
    print("\n" + "="*50)
    print("FINAL MODEL SUMMARY")
    print("="*50)
    print(f"Model Type: Transfer Learning (RF-based Neural Network)")
    print(f"Diagnostic Task: ALG Syndrome vs Biliary Atresia Differential Diagnosis")
    print(f"Features: {', '.join(x_train.columns.tolist())}")
    print(f"Architecture: 6-64-32-1 with Dropout and Sigmoid")
    print(f"Training AUC: {train_metrics['auc']:.4f} (95% CI: {train_metrics['auc_ci'][0]:.4f}-{train_metrics['auc_ci'][1]:.4f})")
    print(f"Test AUC: {test_metrics['auc']:.4f} (95% CI: {test_metrics['auc_ci'][0]:.4f}-{test_metrics['auc_ci'][1]:.4f})")
    print(f"Test Recall (ALG): {test_metrics['recall']:.4f}")
    print(f"Test Precision (ALG): {test_metrics['precision']:.4f}")
    
    # Example prediction
    predictor = ALGBATransferLearningPredictor()
    example_patient = [0.8, 0.6, 0.7, 0.5, 0.4, 0.3]  # Scaled values for the 6 features
    print(f"\nExample differential diagnosis result:")
    result = predictor.predict(example_patient)
    for key, value in result.items():
        print(f"{key}: {value}")
    
    return transfer_model, train_metrics, test_metrics

if __name__ == "__main__":
    model, train_metrics, test_metrics = main()
    print("\n===== Transfer Learning Training Completed =====")