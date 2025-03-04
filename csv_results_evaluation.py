import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import glob

class BERTModelAnalyzer:
    """Analyzes and compares BERT model performance across different configurations."""
    
    def __init__(self, data_dir='.'):
        """Initialize the analyzer with directory containing CSV files."""
        self.data_dir = data_dir
        self.training_metrics = {}
        self.final_metrics = {}
        self.vocab_sizes = [5000, 10000, 20000]
        self.dataset_types = ['single', 'all']
        
    def load_csv_files(self):
        """Load all relevant CSV files from the data directory."""
        # Load training metrics files
        for vocab_size in self.vocab_sizes:
            for is_all_datasets in [False, True]:
                suffix = "_all" if is_all_datasets else ""
                dataset_type = "all" if is_all_datasets else "single"
                config_key = f"vocab_{vocab_size}_{dataset_type}"
                
                # Training metrics file pattern
                train_pattern = f"{self.data_dir}/**/evaluation_metrics_vocab_size_{vocab_size}{suffix}.csv"
                train_files = glob.glob(train_pattern, recursive=True)
                
                if train_files:
                    try:
                        df = pd.read_csv(train_files[0])
                        self.training_metrics[config_key] = df
                        print(f"Loaded training metrics for {config_key}: {len(df)} epochs")
                    except Exception as e:
                        print(f"Error loading {train_files[0]}: {e}")
                
                # Final validation metrics file pattern
                val_pattern = f"{self.data_dir}/**/val_evaluation_metrics_vocab_size_{vocab_size}{suffix}.csv"
                val_files = glob.glob(val_pattern, recursive=True)
                
                if val_files:
                    try:
                        df = pd.read_csv(val_files[0])
                        self.final_metrics[config_key] = df
                        print(f"Loaded final metrics for {config_key}")
                    except Exception as e:
                        print(f"Error loading {val_files[0]}: {e}")
        
        print(f"\nLoaded {len(self.training_metrics)} training metric files")
        print(f"Loaded {len(self.final_metrics)} final validation metric files")
    
    def compare_metrics_over_epochs(self, metric_name, output_file=None):
        """
        Compare a specific metric across epochs for all configurations.
        
        Args:
            metric_name: Name of the metric column in CSV
            output_file: File path to save the plot
        """
        plt.figure(figsize=(12, 7))
        
        line_styles = ['-', '--', '-.', ':']
        markers = ['o', 's', '^', 'D']
        colors = plt.cm.tab10.colors
        
        for i, (config, data) in enumerate(sorted(self.training_metrics.items())):
            if metric_name in data.columns:
                vocab_size = config.split('_')[1]
                dataset_type = "All Datasets" if config.split('_')[2] == 'all' else "Single Dataset"
                label = f"Vocab {vocab_size} ({dataset_type})"
                
                style_idx = i % len(line_styles)
                marker_idx = i % len(markers)
                color_idx = i % len(colors)
                
                plt.plot(data['epoch'], data[metric_name], 
                         linestyle=line_styles[style_idx], 
                         marker=markers[marker_idx], 
                         color=colors[color_idx],
                         linewidth=2, markersize=8, label=label)
        
        plt.title(f"{metric_name.replace('_', ' ').title()} Over Training Epochs", fontsize=15)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        else:
            plt.show()
    
    def compare_final_metrics(self, metric_name, output_file=None):
        """
        Compare final values of a specific metric across all configurations.
        
        Args:
            metric_name: Name of the metric column in CSV
            output_file: File path to save the plot
        """
        vocab_sizes = []
        single_values = []
        all_values = []
        
        for vocab_size in self.vocab_sizes:
            single_key = f"vocab_{vocab_size}_single"
            all_key = f"vocab_{vocab_size}_all"
            
            if single_key in self.final_metrics and metric_name in self.final_metrics[single_key].columns:
                vocab_sizes.append(vocab_size)
                single_values.append(self.final_metrics[single_key][metric_name].iloc[0])
                
                if all_key in self.final_metrics and metric_name in self.final_metrics[all_key].columns:
                    all_values.append(self.final_metrics[all_key][metric_name].iloc[0])
                else:
                    all_values.append(None)
        
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(vocab_sizes))
        width = 0.35
        
        # Plot bars for single dataset
        single_bars = plt.bar(x - width/2, single_values, width, label='Single Dataset', color='#1f77b4')
        
        # Plot bars for all datasets where available
        all_bars = []
        valid_all_values = [v for v in all_values if v is not None]
        if valid_all_values:
            all_bars = plt.bar(x + width/2, all_values, width, label='All Datasets', color='#ff7f0e')
        
        # Add value labels on bars
        for i, v in enumerate(single_values):
            plt.text(i - width/2, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        
        for i, v in enumerate(all_values):
            if v is not None:
                plt.text(i + width/2, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.title(f"Final {metric_name.replace('_', ' ').title()} Comparison", fontsize=15)
        plt.xlabel('Vocabulary Size', fontsize=12)
        plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        plt.xticks(x, vocab_sizes)
        plt.legend()
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        else:
            plt.show()
    
    def analyze_dataset_impact(self, metric_name, output_file=None):
        """
        Analyze the impact of training on all datasets vs single dataset.
        
        Args:
            metric_name: Name of the metric column in CSV
            output_file: File path to save the plot
        """
        vocab_sizes = []
        improvements = []
        
        for vocab_size in self.vocab_sizes:
            single_key = f"vocab_{vocab_size}_single"
            all_key = f"vocab_{vocab_size}_all"
            
            if (single_key in self.final_metrics and all_key in self.final_metrics and
                metric_name in self.final_metrics[single_key].columns and 
                metric_name in self.final_metrics[all_key].columns):
                
                single_value = self.final_metrics[single_key][metric_name].iloc[0]
                all_value = self.final_metrics[all_key][metric_name].iloc[0]
                
                # Calculate percentage improvement
                if 'loss' in metric_name.lower():
                    # For loss metrics, lower is better
                    pct_improvement = ((single_value - all_value) / single_value) * 100
                else:
                    # For other metrics (accuracy, F1, etc.), higher is better
                    pct_improvement = ((all_value - single_value) / single_value) * 100
                
                vocab_sizes.append(vocab_size)
                improvements.append(pct_improvement)
        
        if not vocab_sizes:
            print(f"No data available for impact analysis on {metric_name}")
            return
        
        plt.figure(figsize=(10, 6))
        
        bars = plt.bar(range(len(vocab_sizes)), improvements, color=['g' if i >= 0 else 'r' for i in improvements])
        
        # Add percentage labels
        for i, v in enumerate(improvements):
            plt.text(i, v + (0.5 if v >= 0 else -1.0), f'{v:.2f}%', 
                     ha='center', fontsize=10, 
                     color='black' if v >= 0 else 'red')
        
        plt.title(f"Impact of Using All Datasets on {metric_name.replace('_', ' ').title()}", fontsize=15)
        plt.xlabel('Vocabulary Size', fontsize=12)
        plt.ylabel('Percentage Improvement (%)', fontsize=12)
        plt.xticks(range(len(vocab_sizes)), vocab_sizes)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        else:
            plt.show()
    
    def analyze_vocab_size_impact(self, metric_name, dataset_type='single', output_file=None):
        """
        Analyze the impact of vocabulary size on model performance.
        
        Args:
            metric_name: Name of the metric column in CSV
            dataset_type: 'single' or 'all'
            output_file: File path to save the plot
        """
        vocab_sizes = []
        metric_values = []
        
        for vocab_size in self.vocab_sizes:
            config_key = f"vocab_{vocab_size}_{dataset_type}"
            
            if config_key in self.final_metrics and metric_name in self.final_metrics[config_key].columns:
                vocab_sizes.append(vocab_size)
                metric_values.append(self.final_metrics[config_key][metric_name].iloc[0])
        
        if len(vocab_sizes) < 2:
            print(f"Not enough data points for vocabulary size impact analysis on {metric_name}")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot data points
        plt.plot(vocab_sizes, metric_values, 'o-', linewidth=2, markersize=10, color='#1f77b4')
        
        # Add value labels
        for x, y in zip(vocab_sizes, metric_values):
            plt.annotate(f'{y:.4f}', xy=(x, y), xytext=(0, 10), 
                         textcoords='offset points', ha='center', fontsize=10)
        
        # Add trend line
        z = np.polyfit(vocab_sizes, metric_values, 1)
        p = np.poly1d(z)
        plt.plot(vocab_sizes, p(vocab_sizes), "--", color='#ff7f0e', 
                label=f'Trend: y = {z[0]:.6f}x + {z[1]:.4f}')
        
        dataset_label = "All Datasets" if dataset_type == "all" else "Single Dataset"
        plt.title(f"Effect of Vocabulary Size on {metric_name.replace('_', ' ').title()} ({dataset_label})", 
                 fontsize=15)
        plt.xlabel('Vocabulary Size', fontsize=12)
        plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        else:
            plt.show()
    
    def run_complete_analysis(self, output_dir='bert_analysis'):
        """Run a complete analysis with all comparisons."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Key metrics to analyze
        metrics = ['validation_loss', 'validation_accuracy', 'validation_f1_weighted', 'validation_auc']
        
        # 1. Metrics over epochs
        for metric in metrics:
            self.compare_metrics_over_epochs(metric, f"{output_dir}/{metric}_over_epochs.png")
        
        # 2. Final metrics comparison
        for metric in metrics:
            self.compare_final_metrics(metric, f"{output_dir}/final_{metric}_comparison.png")
        
        # 3. Dataset impact analysis
        for metric in metrics:
            self.analyze_dataset_impact(metric, f"{output_dir}/dataset_impact_{metric}.png")
        
        # 4. Vocabulary size impact analysis
        for metric in metrics:
            for dataset_type in ['single', 'all']:
                if any(f"vocab_{size}_{dataset_type}" in self.final_metrics for size in self.vocab_sizes):
                    self.analyze_vocab_size_impact(
                        metric, dataset_type, 
                        f"{output_dir}/vocab_impact_{metric}_{dataset_type}.png")
        
        print(f"Complete analysis saved to {output_dir}")


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze BERT model evaluation metrics")
    parser.add_argument("--data-dir", default=".", help="Directory containing CSV files")
    parser.add_argument("--output-dir", default="bert_analysis", help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    analyzer = BERTModelAnalyzer(args.data_dir)
    analyzer.load_csv_files()
    analyzer.run_complete_analysis(args.output_dir)