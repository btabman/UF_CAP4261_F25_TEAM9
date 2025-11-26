"""
Main training script for CNN-RNN trajectory prediction model.
"""
from pathlib import Path
import sys
import torch.nn as nn
import torch.optim as optim

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.NFLDataset import NFLDataset
from src.models.CNNRNNHybrid import CNNRNNHybrid
from src.models.CNNRNNHybridTrainer import Trainer

def main():
    """
    CONFIGURATION
    """
    # Declare directories
    DATA_DIR = Path("data/parquet")
    CHECKPOINT_DIR = Path("checkpoints")
    RESULTS_DIR = Path("results")
    
    # Model parameters
    SEQ_LEN = 100
    INPUT_DIM = 4  # s, a, dir, o
    OUTPUT_DIM = 2  # x, y
    
    # CNN-RNN Hybrid config
    CNN_CHANNELS = [64, 128, 128, 64]
    LSTM_HIDDEN = 256
    LSTM_LAYERS = 2
    DROPOUT = 0.3
    
    # Training config
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50
    EARLY_STOP_PATIENCE = 10
    
    # Data split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # For quick testing, limit samples (set to None for full dataset)
    MAX_SAMPLES = None 
    
    print("="*60)
    print("NFL Trajectory Prediction - CNN-RNN Hybrid Model")
    print("="*60)
    
    """
    LOAD AND PREPARE DATA
    """
    print("\nLoading data...")
    dataset = NFLDataset(DATA_DIR, seq_len=SEQ_LEN)
    X, y = dataset.load_and_preprocess(max_samples=MAX_SAMPLES)
    
    print(f"\nCreating train/val/test splits...")
    train_loader, val_loader, test_loader = dataset.create_dataloaders(
        X, y,
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=42
    )
    
    """
    INITIALIZE MODEL
    """
    print(f"\nInitializing CNN-RNN Hybrid model...")
    model = CNNRNNHybrid(
        input_dim=INPUT_DIM,
        cnn_channels=CNN_CHANNELS,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT,
        output_dim=OUTPUT_DIM
    )
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    """
    TRAIN MODEL
    """
    print(f"\nStarting training...")
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        x_max=dataset.x_max,
        y_max=dataset.y_max
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        checkpoint_dir=CHECKPOINT_DIR,
        early_stopping_patience=EARLY_STOP_PATIENCE
    )
    
    """
    EVALUATE TEST SET
    """
    print(f"\nEvaluating on test set...")
    
    # Load best model
    best_model_path = CHECKPOINT_DIR / "cnn_rnn_best_model.pt"
    if best_model_path.exists():
        trainer.load_best_model(best_model_path)
    
    test_results = trainer.test(test_loader)
    
    """
    VISUALIZATION
    """
    print(f"\nGenerating visualizations...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Plot training history
    trainer.plot_training_history(
        save_path=RESULTS_DIR / "cnn_rnn_training_history.png"
    )
    
    # Visualize predictions
    trainer.visualize_predictions(
        predictions=test_results["predictions"],
        targets=test_results["targets"],
        num_examples=3,
        save_path=RESULTS_DIR / "cnn_rnn_test_predictions.png"
    )
    
    """
    PRINT SUMMARY
    """
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation loss: {history['best_val_loss']:.5f}")
    print(f"Test RMSE (unseen data): {test_results['rmse']:.3f} yards")
    print(f"Epochs trained: {history['epochs_trained']}")
    print(f"\nModel saved to: {CHECKPOINT_DIR / 'cnn_rnn_best_model.pt'}")
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Train main CNN-RNN hybrid model
    main()
    