"""
电力系统故障检测项目主脚本
"""

import os
import argparse
import logging
from typing import Dict, Any
import torch

from config import data_config, model_config, training_config, FAULT_TYPES
from data_loader import DataManager
from train import train_model, create_model
from evaluate import evaluate_model, compare_models
from utils import set_seed, setup_logging, plot_training_history, load_model_checkpoint
from models import LSTMModel, CNNModel, EnsembleModel

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Power System Fault Detection')
    
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'compare'], 
                       default='train', help='Mode: train, evaluate, or compare')
    parser.add_argument('--model_type', type=str, choices=['lstm', 'cnn', 'ensemble', 'transformer'],
                       default='ensemble', help='Model type')
    parser.add_argument('--data_dir', type=str, default='dynamic_simulation_datasets',
                       help='Dataset directory')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Results save directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Model checkpoint path for evaluation')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    
    return parser.parse_args()

def setup_directories(save_dir: str) -> Dict[str, str]:
    """设置目录结构"""
    dirs = {
        'main': save_dir,
        'checkpoints': os.path.join(save_dir, 'checkpoints'),
        'logs': os.path.join(save_dir, 'logs'),
        'evaluation': os.path.join(save_dir, 'evaluation'),
        'plots': os.path.join(save_dir, 'plots')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def train_pipeline(args, dirs: Dict[str, str]):
    """训练流水线"""
    logger = logging.getLogger(__name__)
    logger.info("Starting training pipeline...")
    
    # 1. 数据准备
    logger.info("Loading and preparing data...")
    data_manager = DataManager(args.data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = data_manager.prepare_data()
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = data_manager.create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, args.batch_size
    )
    
    # 获取特征维度
    input_dim = data_manager.get_feature_dim()
    logger.info(f"Input dimension: {input_dim}")
    
    # 2. 模型训练
    logger.info(f"Training {args.model_type} model...")
    model, history = train_model(
        train_loader, val_loader, input_dim, 
        args.model_type, dirs['checkpoints']
    )
    
    # 3. 绘制训练历史
    plot_path = os.path.join(dirs['plots'], 'training_history.png')
    plot_training_history(history, plot_path)
    
    # 4. 测试集评估
    logger.info("Evaluating on test set...")
    test_results = evaluate_model(model, test_loader, dirs['evaluation'])
    
    # 5. 保存最终模型
    final_model_path = os.path.join(dirs['checkpoints'], 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': args.model_type,
        'input_dim': input_dim,
        'output_dim': len(FAULT_TYPES),
        'config': model_config.__dict__,
        'test_results': test_results
    }, final_model_path)
    
    logger.info(f"Training completed! Results saved to {dirs['main']}")
    
    return model, test_results

def evaluate_pipeline(args, dirs: Dict[str, str]):
    """评估流水线"""
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation pipeline...")
    
    # 检查checkpoint
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        raise ValueError(f"Checkpoint not found: {args.checkpoint}")
    
    # 1. 数据准备
    logger.info("Loading data...")
    data_manager = DataManager(args.data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = data_manager.prepare_data()
    
    _, _, test_loader = data_manager.create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, args.batch_size
    )
    
    # 2. 加载模型
    logger.info(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    model_type = checkpoint.get('model_type', args.model_type)
    # 如果 checkpoint 里没有 input_dim，就从 DataManager 获取
    input_dim = checkpoint.get('input_dim', None)
    if input_dim is None:
        input_dim = data_manager.get_feature_dim()
        logger.warning(f"'input_dim' not found in checkpoint, using data_manager.get_feature_dim() = {input_dim}")

    # 同理获取 output_dim
    output_dim = checkpoint.get('output_dim', len(FAULT_TYPES))
    if output_dim is None:
        output_dim = len(FAULT_TYPES)
        logger.warning(f"'output_dim' not found in checkpoint, using len(FAULT_TYPES) = {output_dim}")
    model_config_dict = checkpoint.get('config', model_config.__dict__)
    
    model = create_model(model_type, input_dim, output_dim, model_config_dict)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 3. 评估
    logger.info("Evaluating model...")
    results = evaluate_model(model, test_loader, dirs['evaluation'])
    
    logger.info("Evaluation completed!")
    return results

def compare_pipeline(args, dirs: Dict[str, str]):
    """模型比较流水线"""
    logger = logging.getLogger(__name__)
    logger.info("Starting model comparison pipeline...")
    
    # 1. 数据准备
    data_manager = DataManager(args.data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = data_manager.prepare_data()
    
    train_loader, val_loader, test_loader = data_manager.create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, args.batch_size
    )
    
    input_dim = data_manager.get_feature_dim()
    
    # 2. 训练多个模型
    models = {}
    model_types = ['lstm', 'cnn', 'ensemble']
    
    for model_type in model_types:
        logger.info(f"Training {model_type} model...")
        model, _ = train_model(
            train_loader, val_loader, input_dim, 
            model_type, os.path.join(dirs['checkpoints'], model_type)
        )
        models[model_type] = model
    
    # 3. 比较模型
    logger.info("Comparing models...")
    comparison_results = compare_models(models, test_loader, dirs['evaluation'])
    
    logger.info("Model comparison completed!")
    return comparison_results

def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置目录
    dirs = setup_directories(args.save_dir)
    
    # 设置日志
    log_file = os.path.join(dirs['logs'], 'main.log')
    logger = setup_logging(log_file)
    
    # 更新配置
    if args.epochs != 100:
        training_config.epochs = args.epochs
    if args.batch_size != 64:
        training_config.batch_size = args.batch_size
    if args.learning_rate != 0.001:
        training_config.learning_rate = args.learning_rate
    
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        if args.mode == 'train':
            model, results = train_pipeline(args, dirs)
            logger.info(f"Final Hamming Accuracy: {results['hamming_accuracy']:.4f}")
            logger.info(f"Final Subset   Accuracy: {results['subset_accuracy']:.4f}")
            
        elif args.mode == 'evaluate':
            results = evaluate_pipeline(args, dirs)
            logger.info(f"Test Hamming Accuracy: {results['hamming_accuracy']:.4f}")
            logger.info(f"Test Subset   Accuracy: {results['subset_accuracy']:.4f}")

        elif args.mode == 'compare':
            results = compare_pipeline(args, dirs)
            logger.info("Model comparison results:")
            print(results)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()