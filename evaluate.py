import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, multilabel_confusion_matrix, 
                           accuracy_score, precision_recall_fscore_support,
                           hamming_loss, jaccard_score)
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

from config import FAULT_TYPES, training_config
from utils import get_device, plot_confusion_matrix, print_classification_report

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model: nn.Module, device: str = 'auto'):
        self.model = model
        self.device = get_device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """预测"""
        thresholds = torch.tensor(training_config.thresholds, device=self.device)
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                probabilities = torch.sigmoid(outputs)  # 使用sigmoid获取概率
                predictions = (probabilities > thresholds).float()  # 使用动态阈值

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, Any]:
        """评估多标签分类模型"""
        predictions, true_labels, probabilities = self.predict(data_loader)
        
        # 多标签分类指标
        hamming_acc = 1 - hamming_loss(true_labels, predictions)
        subset_acc = accuracy_score(true_labels, predictions)  # 严格准确率
        jaccard = jaccard_score(true_labels, predictions, average='micro')
        
        # 每个类别的指标
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )
        
        # 宏平均和微平均
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='macro', zero_division=0
        )
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='micro', zero_division=0
        )
        
        # 每个类别的指标
        class_metrics = {}
        for i, fault_type in enumerate(FAULT_TYPES):
            class_metrics[fault_type] = {
                'precision': precision[i],
                'recall': recall[i], 
                'f1': f1[i],
                'support': support[i]
            }
        
        results = {
            'hamming_accuracy': hamming_acc,  # 汉明准确率
            'subset_accuracy': subset_acc,    # 子集准确率（所有标签都正确）
            'jaccard_score': jaccard,         # Jaccard指数
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'class_metrics': class_metrics,
            'predictions': predictions,
            'true_labels': true_labels,
            'probabilities': probabilities,
            'multilabel_confusion_matrix': multilabel_confusion_matrix(true_labels, predictions)
        }
        
        return results
    
    def detailed_analysis(self, results: Dict[str, Any], save_dir: str = "evaluation_results"):
        """详细分析多标签分类结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 打印分类报告
        print("=== Multi-Label Classification Report ===")
        print(f"Hamming Accuracy: {results['hamming_accuracy']:.4f}")
        print(f"Subset Accuracy: {results['subset_accuracy']:.4f}")
        print(f"Jaccard Score: {results['jaccard_score']:.4f}")
        print(f"Macro F1: {results['macro_f1']:.4f}")
        print(f"Micro F1: {results['micro_f1']:.4f}")
        
        print("\nPer-class metrics:")
        for fault_type, metrics in results['class_metrics'].items():
            print(f"{fault_type}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
        
        # 绘制每个类别的混淆矩阵
        self._plot_multilabel_confusion_matrices(results, save_dir)
        
        # 保存详细结果
        self._save_multilabel_results_to_csv(results, save_dir)
        self._plot_multilabel_performance(results, save_dir)
        
        logger.info(f"Evaluation results saved to {save_dir}")
        
    def _plot_multilabel_confusion_matrices(self, results: Dict[str, Any], save_dir: str):
        """绘制多标签混淆矩阵"""
        cm_matrices = results['multilabel_confusion_matrix']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (fault_type, cm) in enumerate(zip(FAULT_TYPES, cm_matrices)):
            if i >= len(axes):
                break
                
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{fault_type}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # 隐藏多余的子图
        for i in range(len(FAULT_TYPES), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'multilabel_confusion_matrices.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results_to_csv(self, results: Dict[str, Any], save_dir: str):
        """保存结果到CSV"""
        # 整体指标
        overall_metrics = pd.DataFrame({
            'Metric': ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1',
                      'Micro Precision', 'Micro Recall', 'Micro F1'],
            'Value': [
                results['accuracy'], results['macro_precision'], results['macro_recall'],
                results['macro_f1'], results['micro_precision'], results['micro_recall'],
                results['micro_f1']
            ]
        })
        overall_metrics.to_csv(os.path.join(save_dir, 'overall_metrics.csv'), index=False)
        
        # 类别指标
        class_metrics_df = pd.DataFrame(results['class_metrics']).T
        class_metrics_df.to_csv(os.path.join(save_dir, 'class_metrics.csv'))
        
        # 预测结果
        predictions_df = pd.DataFrame({
            'True_Label': [FAULT_TYPES[i] for i in results['true_labels']],
            'Predicted_Label': [FAULT_TYPES[i] for i in results['predictions']],
            'Correct': results['true_labels'] == results['predictions']
        })
        
        # 添加概率信息
        for i, fault_type in enumerate(FAULT_TYPES):
            predictions_df[f'Prob_{fault_type}'] = results['probabilities'][:, i]
        
        predictions_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    
    def _plot_class_performance(self, results: Dict[str, Any], save_dir: str):
        """绘制各类别性能"""
        class_metrics = results['class_metrics']
        
        metrics = ['precision', 'recall', 'f1']
        fault_types = list(class_metrics.keys())
        
        x = np.arange(len(fault_types))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics):
            values = [class_metrics[ft][metric] for ft in fault_types]
            ax.bar(x + i*width, values, width, label=metric.capitalize())
        
        ax.set_xlabel('Fault Types')
        ax.set_ylabel('Score')
        ax.set_title('Performance by Fault Type')
        ax.set_xticks(x + width)
        ax.set_xticklabels(fault_types, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'class_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_probability_distribution(self, results: Dict[str, Any], save_dir: str):
        """绘制概率分布"""
        probabilities = results['probabilities']
        true_labels = results['true_labels']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, fault_type in enumerate(FAULT_TYPES):
            if i >= len(axes):
                break
                
            # 正确预测的概率分布
            correct_mask = true_labels == i
            correct_probs = probabilities[correct_mask, i] if np.any(correct_mask) else []
            
            # 错误预测的概率分布
            incorrect_mask = (true_labels != i) & (results['predictions'] == i)
            incorrect_probs = probabilities[incorrect_mask, i] if np.any(incorrect_mask) else []
            
            axes[i].hist(correct_probs, bins=20, alpha=0.7, label='Correct', density=True)
            axes[i].hist(incorrect_probs, bins=20, alpha=0.7, label='Incorrect', density=True)
            axes[i].set_title(f'{fault_type}')
            axes[i].set_xlabel('Probability')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'probability_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _save_multilabel_results_to_csv(self, results: Dict[str, Any], save_dir: str):
        """保存多标签分类结果到 CSV"""
        # 1. 保存整体指标
        overall = {
            'hamming_accuracy': results['hamming_accuracy'],
            'subset_accuracy': results['subset_accuracy'],
            'jaccard_score': results['jaccard_score'],
            'macro_precision': results['macro_precision'],
            'macro_recall': results['macro_recall'],
            'macro_f1': results['macro_f1'],
            'micro_precision': results['micro_precision'],
            'micro_recall': results['micro_recall'],
            'micro_f1': results['micro_f1'],
        }
        pd.DataFrame.from_dict(overall, orient='index', columns=['value']) \
          .to_csv(os.path.join(save_dir, 'overall_multilabel_metrics.csv'), header=True)

        # 2. 保存每类指标
        class_metrics_df = pd.DataFrame(results['class_metrics']).T
        class_metrics_df.to_csv(os.path.join(save_dir, 'class_multilabel_metrics.csv'))

        # 3. 保存预测明细
        preds = results['predictions']
        trues = results['true_labels']
        probs = results['probabilities']
        cols = []
        for fault in FAULT_TYPES:
            cols += [f'pred_{fault}', f'true_{fault}', f'prob_{fault}']
        data = np.hstack([
            preds,    # multi-hot 0/1
            trues,    # multi-hot 0/1
            probs     # 概率
        ])
        df = pd.DataFrame(data, columns=cols)
        df.to_csv(os.path.join(save_dir, 'predictions_multilabel.csv'), index=False)

    def _plot_multilabel_performance(self, results: Dict[str, Any], save_dir: str):
        """绘制每个故障类别的 Precision/Recall/F1 条形图"""
        class_metrics = results['class_metrics']
        faults = list(class_metrics.keys())
        precisions = [class_metrics[f]['precision'] for f in faults]
        recalls    = [class_metrics[f]['recall']    for f in faults]
        f1s        = [class_metrics[f]['f1']        for f in faults]

        x = np.arange(len(faults))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precisions, width, label='Precision')
        ax.bar(x,         recalls,    width, label='Recall')
        ax.bar(x + width, f1s,        width, label='F1-score')

        ax.set_xticks(x)
        ax.set_xticklabels(faults, rotation=45)
        ax.set_ylabel('Score')
        ax.set_title('Per-class Precision / Recall / F1')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'multilabel_class_performance.png'), dpi=300)
        plt.close()

def evaluate_model(model: nn.Module, test_loader: DataLoader, 
                  save_dir: str = "evaluation_results") -> Dict[str, Any]:
    """评估模型的主函数（多标签版）"""
    evaluator = ModelEvaluator(model)
    results = evaluator.evaluate(test_loader)
    evaluator.detailed_analysis(results, save_dir)
    
    # 打印 summary——多标签主要关注 Hamming Accuracy & Subset Accuracy
    logger.info("=== Evaluation Summary ===")
    logger.info(f"Hamming Accuracy: {results['hamming_accuracy']:.4f}")
    logger.info(f"Subset Accuracy: {results['subset_accuracy']:.4f}")
    logger.info(f"Jaccard Score : {results['jaccard_score']:.4f}")
    logger.info(f"Macro F1      : {results['macro_f1']:.4f}")
    logger.info(f"Micro F1      : {results['micro_f1']:.4f}")
    
    return results

def compare_models(models: Dict[str, nn.Module], test_loader: DataLoader, 
                  save_dir: str = "model_comparison") -> pd.DataFrame:
    """比较多个模型"""
    os.makedirs(save_dir, exist_ok=True)
    
    comparison_results = []
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        evaluator = ModelEvaluator(model)
        results = evaluator.evaluate(test_loader)
        
        comparison_results.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Macro_Precision': results['macro_precision'],
            'Macro_Recall': results['macro_recall'],
            'Macro_F1': results['macro_f1'],
            'Micro_F1': results['micro_f1'],
            'Parameters': model.get_num_parameters(),
            'Model_Size_MB': model.get_model_size()
        })
    
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(os.path.join(save_dir, 'model_comparison.csv'), index=False)
    
    # 绘制比较图
    metrics = ['Accuracy', 'Macro_F1', 'Micro_F1']
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        axes[i].bar(comparison_df['Model'], comparison_df[metric])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Model comparison saved to {save_dir}")
    return comparison_df

