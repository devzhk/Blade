'''
This script visualizes how CRPS, SSR, and relative L2 metrics evolve over Blade iterations.
It loads intermediate results from blade_iter_{i}_x.pt files and computes metrics against ground truth.
'''
import os
from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import metric functions from eval.py
from eval import relative_l2, crps, get_spread2, get_skill2


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config):
    """
    Main function to plot metric evolution over Blade iterations.
    
    Args:
        config: Hydra configuration object
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parse experiment directories and output name from config
    # Support multiple experiment directories for different noise levels
    exp_dirs = config.get('exp_dirs', None)
    if exp_dirs is None:
        exp_dir = config.get('exp_dir', 'exps/noise-1')
        exp_dirs = [exp_dir]
        exp_labels = ['noise=1']
    else:
        # exp_dirs should be a list like ['exps/noise-0', 'exps/noise-1', 'exps/noise-2']
        if isinstance(exp_dirs, str):
            exp_dirs = [exp_dirs]
        # Extract labels from directory names (e.g., 'exps/noise-1' -> 'noise=1')
        exp_labels = [Path(d).name.replace('noise-', 'noise=') for d in exp_dirs]
    
    output_name = config.get('output_name', 'blade_evolution.png')
    output_dir = config.get('output_dir', exp_dirs[0])
    
    print(f"Experiment directories: {exp_dirs}")
    print(f"Labels: {exp_labels}")
    print(f"Output directory: {output_dir}")
    print(f"Output file: {output_name}")
    
    # Instantiate forward operator and dataset using Hydra
    forward_op = instantiate(config.problem.model, device=device)
    testset = instantiate(config.problem.data)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    
    print(f"Loaded dataset with {len(testset)} samples")
    print(f"Test case IDs: {testset.id_list}")
    
    # Storage for all experiments
    all_results = {}
    
    # Process each experiment directory
    for exp_dir, exp_label in zip(exp_dirs, exp_labels):
        print(f"\n{'='*60}")
        print(f"Processing: {exp_label} ({exp_dir})")
        print(f"{'='*60}")
        
        # Determine number of iterations by checking one test case directory
        test_case_0_dir = Path(exp_dir) / str(testset.id_list[0])
        if not test_case_0_dir.exists():
            print(f"Warning: Test case directory not found: {test_case_0_dir}, skipping...")
            continue
        
        # Find all blade_iter_*_x.pt files
        iter_files = sorted(test_case_0_dir.glob('blade_iter_*_x.pt'))
        num_iterations = len(iter_files)
        print(f"Found {num_iterations} iterations")
        
        if num_iterations == 0:
            print(f"Warning: No blade_iter_*_x.pt files found in {test_case_0_dir}, skipping...")
            continue
        
        # Initialize storage for metrics across all test cases and iterations
        # Shape: [num_iterations, num_test_cases]
        crps_values = np.zeros((num_iterations, len(testset)))
        spread2_values = np.zeros((num_iterations, len(testset)))
        skill2_values = np.zeros((num_iterations, len(testset)))
        rel_l2_values = np.zeros((num_iterations, len(testset)))
        
        # Loop over all test cases
        for i, data in enumerate(tqdm(testloader, desc=f"Processing test cases for {exp_label}")):
            # Handle data loading following main.py pattern (lines 85-92)
            if isinstance(data, torch.Tensor):
                data = data.to(device)
            elif isinstance(data, dict):
                assert 'target' in data.keys(), "'target' must be in the data dict"
                for key, val in data.items():
                    if isinstance(val, torch.Tensor):
                        data[key] = val.to(device)
            
            test_id = testset.id_list[i]
            
            # Load ground truth target
            target = data['target'].squeeze(0)  # Remove batch dimension (1, C, H, W) -> (C, H, W)
            
            # Unnormalize target
            target_unnorm = forward_op.unnormalize(target)
            
            # Loop over all iterations
            for iter_idx in range(num_iterations):
                # Load particles from blade_iter_{iter_idx}_x.pt
                particles_path = Path(exp_dir) / str(test_id) / f'blade_iter_{iter_idx}_x.pt'
                
                if not particles_path.exists():
                    print(f"Warning: File not found: {particles_path}, skipping...")
                    continue
                
                # Load particles
                particles_data = torch.load(particles_path, map_location=device)
                particles = particles_data['particles']  # Shape: (num_particles, C, H, W)
                
                # Unnormalize particles
                particles_unnorm = forward_op.unnormalize(particles)
                
                # Reshape to match metric function signatures
                # predictions: (N, M, ...) where N=1 (batch), M=num_particles
                # targets: (N, ...)
                pred = particles_unnorm.unsqueeze(0)  # (1, num_particles, C, H, W)
                tgt = target_unnorm.unsqueeze(0)     # (1, C, H, W)
                
                # Compute metrics
                crps_val = crps(pred, tgt).item()
                spread2 = get_spread2(pred, tgt).item()
                skill2 = get_skill2(pred, tgt).item()
                rel_l2_val = relative_l2(pred, tgt).item()
                
                # Store values
                crps_values[iter_idx, i] = crps_val
                spread2_values[iter_idx, i] = spread2
                skill2_values[iter_idx, i] = skill2
                rel_l2_values[iter_idx, i] = rel_l2_val
        
        # Compute mean across test cases
        # Following eval.py logic: SSR is computed from the MEAN of spread2 and skill2
        crps_mean = crps_values.mean(axis=1)
        spread2_mean = spread2_values.mean(axis=1)
        skill2_mean = skill2_values.mean(axis=1)
        
        # Compute SSR from aggregated spread2 and skill2 (following eval.py line 322)
        ssr_mean = np.sqrt(spread2_mean) / np.sqrt(skill2_mean)
        rel_l2_mean = rel_l2_values.mean(axis=1)
        
        iterations = np.arange(num_iterations)
        
        # Store results
        all_results[exp_label] = {
            'iterations': iterations,
            'crps_mean': crps_mean,
            'ssr_mean': ssr_mean,
            'rel_l2_mean': rel_l2_mean,
        }
        
        print(f"Final CRPS:   {crps_mean[-1]:.4f}")
        print(f"Final SSR:    {ssr_mean[-1]:.4f}")
        print(f"Final Rel L2: {rel_l2_mean[-1]:.4f}")
    
    # Create plot with 3 subplots matching the style of the reference image
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Define colors for different noise levels (matching the image style)
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c']  # blue, orange, green
    markers = ['s', 'o', '^']  # circle, square, triangle
    
    # Plot CRPS
    for idx, (label, results) in enumerate(all_results.items()):
        axes[0].plot(results['iterations'], results['crps_mean'], 
                    color=colors[idx], marker=markers[idx], linewidth=2, 
                    markersize=5, label=label)
    axes[0].set_xlabel('K (Number of iterations)', fontsize=12)
    axes[0].set_ylabel('CRPS (↓)', fontsize=12)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(fontsize=10)
    
    # Plot Relative L2
    for idx, (label, results) in enumerate(all_results.items()):
        axes[1].plot(results['iterations'], results['rel_l2_mean'], 
                    color=colors[idx], marker=markers[idx], linewidth=2, 
                    markersize=5, label=label)
    axes[1].set_xlabel('K (Number of iterations)', fontsize=12)
    axes[1].set_ylabel('Relative L2 error (↓)', fontsize=12)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(fontsize=10)
    
    # Plot SSR
    for idx, (label, results) in enumerate(all_results.items()):
        axes[2].plot(results['iterations'], results['ssr_mean'], 
                    color=colors[idx], marker=markers[idx], linewidth=2, 
                    markersize=5, label=label)
    # Add horizontal line at y=1 for perfect calibration
    axes[2].axhline(y=1.0, color='k', linestyle='--', linewidth=1.5, 
                   alpha=0.7, label='Perfect calibration')
    axes[2].set_xlabel('K (Number of iterations)', fontsize=12)
    axes[2].set_ylabel('Spread-Skill ratio (→1)', fontsize=12)
    axes[2].set_ylim(0, 1.0)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / output_name
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"Saved plot to {output_path}")
    print(f"{'='*60}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    for label, results in all_results.items():
        print(f"\n{label}:")
        print(f"  Final CRPS:   {results['crps_mean'][-1]:.4f}")
        print(f"  Final SSR:    {results['ssr_mean'][-1]:.4f}")
        print(f"  Final Rel L2: {results['rel_l2_mean'][-1]:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()

