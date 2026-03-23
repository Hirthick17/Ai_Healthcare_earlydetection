"""
src/verify_extraction.py
Validates the extracted 12-lead ECG signal array.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    npy_path = sys.argv[1] if len(sys.argv) > 1 else "synthetic_extracted.npy"
        
    print(f"Loading {npy_path}...")
    try:
        data = np.load(npy_path)
    except Exception as e:
        print(f"Failed to load: {e}")
        sys.exit(1)
        
    print(f"Shape: {data.shape}")
    assert data.shape == (1000, 12), f"Expected shape (1000, 12), got {data.shape}"
    
    dead_leads = []
    lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    
    # Dead Lead Check
    for i in range(12):
        var = np.var(data[:, i])
        if var == 0:
            dead_leads.append(lead_names[i])
            
    if dead_leads:
        raise ValueError(f"Dead Lead Check Failed. The following leads are flatlines (variance=0): {', '.join(dead_leads)}")
        
    print("Dead Lead Check Passed for all 12 channels.")
    
    # Plotting
    print("Generating proof plot...")
    fig, axes = plt.subplots(12, 1, figsize=(10, 16), sharex=True)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
        
    for i in range(12):
        axes[i].plot(data[:, i], color='black', linewidth=1.5)
        axes[i].set_ylabel(lead_names[i], rotation=0, labelpad=20, va='center')
        axes[i].grid(True, linestyle='--', alpha=0.6)
        
    axes[-1].set_xlabel("Time steps")
    fig.suptitle("Extracted 12-Lead ECG Signals", fontsize=16)
    plt.tight_layout()
    
    out_img = "extraction_proof.png"
    plt.savefig(out_img, dpi=150)
    print(f"Saved {out_img} successfully.")
    
if __name__ == "__main__":
    main()
