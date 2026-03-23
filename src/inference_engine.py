"""
src/inference_engine.py
=======================
AI Inference Router for 12-lead ECG NumPy arrays.

Loads an extracted (1000, 12) trace, converts to a PyTorch tensor,
runs it through the DenseNet1D model, and outputs a structured JSON
artifact of the 5-class PTB-XL predictions for the dashboard engine.

Classes: ["NORM", "MI", "STTC", "CD", "HYP"]
"""

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Import the DenseNet1D architecture per the project structure
from src.densenet1d import DenseNet1D

PTBXL_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def main():
    parser = argparse.ArgumentParser(description="AI Inference Router for 12-lead ECG")
    parser.add_argument("--input", type=str, required=True, help="Path to the extracted .npy array (e.g. synthetic_extracted.npy)")
    parser.add_argument("--weights", type=str, required=True, help="Path to the pretrained model weights (.pt or .pth)")
    parser.add_argument("--out", type=str, default="ai_prediction.json", help="Path to save the JSON output artifact")
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Step 1: Model Initialization
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/4] Initialising DenseNet1D on {device}...")
    
    # Initialize the architecture with 5 output classes mapping to PTBXL_CLASSES
    model = DenseNet1D(in_channels=12, num_classes=len(PTBXL_CLASSES))
    
    try:
        # Load pre-trained weights
        model.load_state_dict(torch.load(args.weights, map_location=device))
    except Exception as e:
        print(f"Error loading weights at {args.weights}:\n{e}")
        print("\nPlease ensure you are passing a valid weights file configured for 5 classes.")
        return
        
    model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # Step 2: Data Ingestion & Transposition (CRITICAL)
    # ---------------------------------------------------------
    target_shape = (1000, 12)
    print(f"[2/4] Loading and transposing input array: {args.input}...")
    try:
        ecg_data = np.load(args.input)
    except Exception as e:
        print(f"Error loading input .npy:\n{e}")
        return

    # Enforce shape requirements
    if ecg_data.shape != target_shape:
        if ecg_data.shape == (12, 1000):
            print(f"      Got shape {ecg_data.shape}. Transposing to {target_shape}.")
            ecg_data = ecg_data.T
        else:
            print(f"Error: Expected input shape {target_shape} or (12, 1000), got {ecg_data.shape}.")
            return

    # Convert to PyTorch FloatTensor
    tensor_data = torch.from_numpy(ecg_data).float()
    
    # Transpose and add Batch dimension: (1000, 12) -> (12, 1000) -> (1, 12, 1000)
    tensor_data = tensor_data.transpose(0, 1).unsqueeze(0)
    tensor_data = tensor_data.to(device)

    # ---------------------------------------------------------
    # Step 3: Forward Pass & Post-Processing
    # ---------------------------------------------------------
    print("[3/4] Running forward pass through DenseNet1D...")
    with torch.no_grad():
        logits = model(tensor_data)
        # Apply Softmax to get probabilities
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # Map the highest probability index
    pred_idx = int(np.argmax(probs))
    predicted_class = PTBXL_CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    probabilities_dict = {
        cls_name: round(float(prob), 4) 
        for cls_name, prob in zip(PTBXL_CLASSES, probs)
    }

    # ---------------------------------------------------------
    # Step 4: Output Artifact Generation
    # ---------------------------------------------------------
    print(f"[4/4] Finalizing Output Details for Class: {predicted_class}")
    
    output_data = {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": probabilities_dict
    }

    print("\n==================================")
    print("      AI INFERENCE RESULTS        ")
    print("==================================")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence     : {confidence * 100:.2f}%\n")
    print("Probability Distribution:")
    for k, v in probabilities_dict.items():
        print(f"  - {k:<5}: {v:.4f}")

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print("\n==================================")
    print(f"Artifact successfully saved to: {out_path.name}")


if __name__ == "__main__":
    main()
