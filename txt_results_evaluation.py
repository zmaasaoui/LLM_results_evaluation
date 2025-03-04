import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

class BERTMetricsAnalyzer:
    """Class for analyzing and visualizing BERT model evaluation metrics."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.evaluation_metrics = {}
        self.final_metrics = {}
        self.vocab_sizes = [5000, 10000, 20000]
        self.dataset_types = ['single', 'all']
        
    def parse_evaluation_file(self, filename):
        """
        Parse an evaluation metrics file to extract metrics for each epoch.
        
        Args:
            filename: Path to the evaluation metrics file
            
        Returns:
            DataFrame containing metrics for each epoch
        """
        try:
            with open(filename, 'r') as file:
                content = file.read()
                
            # Extract epoch metrics using regex
            pattern = r"Epoch (\d+) Metrics:\nTraining Loss: ([\d\.]+)\nTraining Accuracy: ([\d\.]+)\nTraining F1 \(Weighted\): ([\d\.]+).*?\nValidation Loss: ([\d\.]+)\nValidation Accuracy: ([\d\.]+)\nValidation F1 \(Weighted\): ([\d\.]+)"
            matches = re.findall(pattern, content, re.DOTALL)
            
            if not matches:
                print(f"No metrics found in {filename}")
                return None
            
            # Create DataFrame from matched metrics
            data = []
            for match in matches:
                epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1 = match
                data.append({
                    'Epoch': int(epoch),
                    'Training Loss': float(train_loss),
                    'Training Accuracy': float(train_acc),
                    'Training F1': float(train_f1),
                    'Validation Loss': float(val_loss),
                    'Validation Accuracy': float(val_acc),
                    'Validation F1': float(val_f1)
                })
            
            return pd.DataFrame(data)
        
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            return None
    
    def parse_final_metrics_file(self, filename):
        """
        Parse a final validation metrics file.
        
        Args:
            filename: Path to the final validation metrics file
            
        Returns:
            Dictionary containing final metrics
        """
        try:
            with open(filename, 'r') as file:
                content = file.read()
            
            # Extract key metrics
            metrics = {}
            
            # Get final validation loss
            loss_match = re.search(r"Val Final Validation Loss: ([\d\.]+)", content)
            if loss_match:
                metrics['Validation Loss'] = float(loss_match.group(1))
            
            # Get F1 score
            f1_match = re.search(r"Val F1 Score \(Weighted\): ([\d\.]+)", content)
            if f1_match:
                metrics['F1 Score'] = float(f1_match.group(1))
            
            # Get accuracy
            acc_match = re.search(r"Validation Accuracy: ([\d\.]+)", content)
            if acc_match:
                metrics['Accuracy'] = float(acc_match.group(1))
            
            # Get AUC
            auc_match = re.search(r"AUC: ([\d\.]+)", content)
            if auc_match:
                metrics['AUC'] = float(auc_match.group(1))
                
            # Extract per-class AUC values
            auc_section = re.search(r"Per-class AUC:(.*?)Confusion Matrix:", content, re.DOTALL)
            if auc_section:
                class_pattern = r"  (\w+): ([\d\.]+)"
                class_matches = re.findall(class_pattern, auc_section.group(1))
                for class_name, auc_value in class_matches:
                    metrics[f'Class_{class_name}'] = float(auc_value)
            
            # Extract confusion matrix
            matrix_match = re.search(r"Confusion Matrix:(.*?)$", content, re.DOTALL | re.MULTILINE)
            if matrix_match:
                matrix_text = matrix_match.group(1).strip()
                lines = matrix_text.split('\n')
                matrix = []
                for line in lines:
                    row = [int(x) for x in line.split()]
                    if row:  # Skip empty lines
                        matrix.append(row)
                
                metrics['Confusion Matrix'] = np.array(matrix)
            
            return metrics
            
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            return None
    
    def load_all_files(self, directory='./'):
        """
        Load all metrics files from the specified directory.
        
        Args:
            directory: Directory containing the metrics files
        """
        # Process evaluation metrics files
        for vocab_size in self.vocab_sizes:
            for dataset_type in self.dataset_types:
                suffix = "_all" if dataset_type == 'all' else ""
                filename = f"{directory}/evaluation_metrics_vocab_size_{vocab_size}{suffix}.txt"
                
                if os.path.exists(filename):
                    config_key = f"vocab_{vocab_size}_{dataset_type}"
                    self.evaluation_metrics[config_key] = self.parse_evaluation_file(filename)
                    
                    # Also load final metrics
                    final_filename = f"{directory}/final_validation_metrics_vocab_size_{vocab_size}{suffix}.txt"
                    if os.path.exists(final_filename):
                        self.final_metrics[config_key] = self.parse_final_metrics_file(final_filename)
    
    def plot_metrics_over_epochs(self, metric_name, title, ylabel, output_file=None):
        """
        Plot a specific metric over epochs for all configurations.
        
        Args:
            metric_name: Name of the metric to plot
            title: Plot title
            ylabel: Label for y-axis
            output_file: File to save the plot to (optional)
        """
        plt.figure(figsize=(12, 7))
        
        # Different line styles and markers for better differentiation
        line_styles = ['-', '--', '-.', ':']
        markers = ['o', 's', '^', 'D', 'v', '<', '>']
        colors = plt.cm.tab10.colors
        
        legend_entries = []
        for i, (config, data) in enumerate(sorted(self.evaluation_metrics.items())):
            if data is not None and metric_name in data.columns:
                vocab_size = config.split('_')[1]
                dataset_type = "All Datasets" if config.split('_')[2] == 'all' else "Single Dataset"
                label = f"Vocab Size {vocab_size} ({dataset_type})"
                
                line_style = line_styles[i % len(line_styles)]
                marker = markers[i % len(markers)]
                color = colors[i % len(colors)]
                
                plt.plot(data['Epoch'], data[metric_name], 
                         linestyle=line_style, marker=marker, color=color,
                         linewidth=2, markersize=8, label=label)
                
                legend_entries.append(label)
        
        plt.title(title, fontsize=15)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(True, alpha=0.3)
        
        # Add legend with custom ordering
        if legend_entries:
            plt.legend(fontsize=10, loc='best')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        
        plt.show()
    
    def plot_final_metrics_comparison(self, metric_name, title, ylabel, output_file=None):
        """
        Create bar chart comparing final metrics across configurations.
        
        Args:
            metric_name: Name of the metric to compare
            title: Plot title
            ylabel: Label for y-axis
            output_file: File to save the plot to (optional)
        """
        # Extract relevant data
        vocab_sizes = []
        single_dataset_values = []
        all_datasets_values = []
        
        for config, metrics in sorted(self.final_metrics.items()):
            if metrics and metric_name in metrics:
                vocab_size = int(config.split('_')[1])
                dataset_type = config.split('_')[2]
                
                if dataset_type == 'single':
                    single_dataset_values.append((vocab_size, metrics[metric_name]))
                else:
                    all_datasets_values.append((vocab_size, metrics[metric_name]))
                
                if vocab_size not in vocab_sizes:
                    vocab_sizes.append(vocab_size)
        
        # Sort by vocab size
        vocab_sizes.sort()
        single_dataset_values.sort(key=lambda x: x[0])
        all_datasets_values.sort(key=lambda x: x[0])
        
        # Prepare for plotting
        single_vals = [val for size, val in single_dataset_values if size in vocab_sizes]
        all_vals = [val for size, val in all_datasets_values if size in vocab_sizes]
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(vocab_sizes))
        width = 0.35
        
        plt.bar(x - width/2, single_vals, width, label='Single Dataset', color='#1f77b4', alpha=0.8)
        plt.bar(x + width/2, all_vals, width, label='All Datasets', color='#ff7f0e', alpha=0.8)
        
        # Add value labels
        for i, v in enumerate(single_vals):
            plt.text(i - width/2, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        
        for i, v in enumerate(all_vals):
            plt.text(i + width/2, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.title(title, fontsize=15)
        plt.xlabel('Vocabulary Size', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(x, vocab_sizes)
        plt.legend()
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        
        plt.show()
    
    def plot_per_class_auc(self, output_file=None):
        """
        Create a heatmap showing per-class AUC values for each configuration.
        
        Args:
            output_file: File to save the plot to (optional)
        """
        # Extract class names and AUC values
        class_names = set()
        for metrics in self.final_metrics.values():
            if metrics:
                for key in metrics.keys():
                    if key.startswith('Class_'):
                        class_names.add(key.replace('Class_', ''))
        
        class_names = sorted(list(class_names))
        
        # Create a DataFrame for the heatmap
        configs = []
        data = []
        
        for config, metrics in sorted(self.final_metrics.items()):
            if metrics:
                vocab_size = config.split('_')[1]
                dataset_type = "All Datasets" if config.split('_')[2] == 'all' else "Single Dataset"
                config_name = f"Vocab {vocab_size} ({dataset_type})"
                
                row = []
                for class_name in class_names:
                    key = f'Class_{class_name}'
                    if key in metrics:
                        row.append(metrics[key])
                    else:
                        row.append(np.nan)
                
                configs.append(config_name)
                data.append(row)
        
        # Create heatmap
        plt.figure(figsize=(14, len(configs) * 0.8))
        
        df = pd.DataFrame(data, index=configs, columns=class_names)
        sns.heatmap(df, annot=True, fmt='.4f', cmap='YlGnBu', linewidths=0.5, 
                    vmin=0.9, vmax=1.0, cbar_kws={'label': 'AUC Value'})
        
        plt.title('Per-Class AUC Values by Model Configuration', fontsize=15)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        
        plt.show()
    
    def plot_confusion_matrix(self, config_key, output_file=None):
        """
        Plot confusion matrix for a specific configuration.
        
        Args:
            config_key: Configuration key
            output_file: File to save the plot to (optional)
        """
        if config_key in self.final_metrics and 'Confusion Matrix' in self.final_metrics[config_key]:
            vocab_size = config_key.split('_')[1]
            dataset_type = "All Datasets" if config_key.split('_')[2] == 'all' else "Single Dataset"
            title = f"Confusion Matrix - Vocab Size {vocab_size} ({dataset_type})"
            
            matrix = self.final_metrics[config_key]['Confusion Matrix']
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
            
            # Try to determine class names
            class_names = []
            for key in self.final_metrics[config_key].keys():
                if key.startswith('Class_'):
                    class_names.append(key.replace('Class_', ''))
            
            if class_names:
                plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45, ha='right')
                plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
            
            plt.title(title, fontsize=15)
            plt.xlabel('Predicted', fontsize=12)
            plt.ylabel('True', fontsize=12)
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {output_file}")
            
            plt.show()
    
    def analyze_all_metrics(self, output_dir='plots'):
        """
        Analyze all metrics and create visualizations.
        
        Args:
            output_dir: Directory to save plots to
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Plot validation metrics over epochs
        self.plot_metrics_over_epochs('Validation Loss', 
                                     'Validation Loss Over Epochs', 
                                     'Loss',
                                     f'{output_dir}/validation_loss_over_epochs.png')
        
        self.plot_metrics_over_epochs('Validation Accuracy', 
                                     'Validation Accuracy Over Epochs', 
                                     'Accuracy',
                                     f'{output_dir}/validation_accuracy_over_epochs.png')
        
        self.plot_metrics_over_epochs('Validation F1', 
                                     'Validation F1 Score Over Epochs', 
                                     'F1 Score',
                                     f'{output_dir}/validation_f1_over_epochs.png')
        
        # 2. Compare final metrics
        self.plot_final_metrics_comparison('F1 Score', 
                                         'Final F1 Score Comparison', 
                                         'F1 Score',
                                         f'{output_dir}/final_f1_comparison.png')
        
        self.plot_final_metrics_comparison('Accuracy', 
                                         'Final Accuracy Comparison', 
                                         'Accuracy',
                                         f'{output_dir}/final_accuracy_comparison.png')
        
        self.plot_final_metrics_comparison('AUC', 
                                         'Final AUC Comparison', 
                                         'AUC',
                                         f'{output_dir}/final_auc_comparison.png')
        
        # 3. Plot per-class AUC values
        self.plot_per_class_auc(f'{output_dir}/per_class_auc.png')
        
        # 4. Plot confusion matrices for each configuration
        for config_key in self.final_metrics.keys():
            vocab_size = config_key.split('_')[1]
            dataset_type = config_key.split('_')[2]
            filename = f'{output_dir}/confusion_matrix_vocab_{vocab_size}_{dataset_type}.png'
            self.plot_confusion_matrix(config_key, filename)
        
        print(f"All plots saved to {output_dir}")


def main():
    """Entry point for the script."""
    analyzer = BERTMetricsAnalyzer()
    
    # Load metrics from files
    analyzer.load_all_files()
    
    # Analyze and create visualizations
    analyzer.analyze_all_metrics()


if __name__ == "__main__":
    main()