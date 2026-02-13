# new_eval.py
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.calibration import calibration_curve

# Import model and dataset
from priorViT import PriorViT
from train_vit_classifier import GeneratedDataset 

def calculate_ece(y_true, y_prob, n_bins=10):
    """
    Calculates the Expected Calibration Error (ECE).
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        # Find samples that fall into the current bin
        bin_mask = (y_prob > bins[i]) & (y_prob <= bins[i+1])
        bin_size = np.sum(bin_mask)
        
        if bin_size > 0:
            # Accuracy in this bin: fraction of positive labels
            bin_acc = np.mean(y_true[bin_mask])
            # Confidence in this bin: average predicted probability
            bin_conf = np.mean(y_prob[bin_mask])
            # Weighted absolute difference
            ece += (bin_size / len(y_true)) * np.abs(bin_acc - bin_conf)
            
    return ece

def plot_reliability_diagram(y_true, y_prob, n_bins=10, title="Reliability Diagram (GAN Baseline)"):
    """
    Plots the Reliability Diagram (Confidence vs. Accuracy).
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated", color="gray")
    plt.plot(prob_pred, prob_true, marker="o", label="GAN Baseline", color="red")
    
    # Fill the area to show over/under confidence
    plt.fill_between(prob_pred, prob_pred, prob_true, color='red', alpha=0.1, label="Calibration Gap")
    
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Fraction of Positives (Accuracy)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("calibration_diagram_gan.png")
    print(f"Reliability diagram saved to calibration_diagram_gan.png")
    plt.show()

def evaluate():
    CONFIG = {
        "gen_test_dir": "Generated_Test_M36", 
        "csv_file": "../5_folds_split_3D/fold_1_test.csv",          
        "atlas_path": "npy/disease_atlas.npy",      
        "model_path": "model_vit/prior_vit_latest.pth",    
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    DEVICE = torch.device(CONFIG["device"])

    # 1. Load Data
    print("Loading Test Data...")
    dataset = GeneratedDataset(CONFIG["gen_test_dir"], CONFIG["csv_file"])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # 2. Load Model & Atlas
    print(f"Loading Model...")
    atlas = np.load(CONFIG['atlas_path'])
    model = PriorViT(atlas=atlas).to(DEVICE)
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=DEVICE))
    model.eval()

    # 3. Evaluation Loop
    all_preds = []
    all_labels = []
    all_probs = []

    print("Running Inference...")
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(imgs) 
            probs = torch.softmax(outputs, dim=1) 
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1]) 

    # 4. Standard Metrics
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    # 5. Calibration Metrics
    ece_score = calculate_ece(all_labels, all_probs, n_bins=10)
    
    print("\n" + "="*40)
    print("FINAL EVALUATION RESULTS (GAN BASELINE)")
    print("="*40)
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"AUC Score: {auc:.4f}")
    print(f"ECE (Expected Calibration Error): {ece_score:.4f}")
    print("-" * 40)
    
    # 6. Visualization
    plot_reliability_diagram(all_labels, all_probs)

if __name__ == "__main__":
    evaluate()