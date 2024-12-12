import torch
import os
import argparse
import 심신개_test.evaluate_func as evaluate_func
import logging

from custom_datasets import get_images, get_dataset, get_data_loaders
from model import load_model
from config import ALL_CLASSES, LABEL_COLORS_LIST
from engine import train, validate
from utils import save_model, SaveBestModel, save_plots, SaveBestModelIOU
from torch.optim.lr_scheduler import MultiStepLR

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Set up logging
logger = logging.getLogger('TrainingLogger')
logger.setLevel(logging.INFO)
# Create a file handler to save logs to a file
log_file = 'training_log_large.txt'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
# Create a console handler to log to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Log some starting information
logger.info(f"Starting training with seed: {seed}")

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument(
    '--epochs',
    default=30,
    help='number of epochs to train for',
    type=int
)
parser.add_argument(
    '--lr',
    default=0.0001,
    help='learning rate for optimizer',
    type=float
)
parser.add_argument(
    '--batch',
    default=4,
    help='batch size for data loader',
    type=int
)
parser.add_argument(
    '--imgsz', 
    default=[512, 512],
    type=int,
    nargs='+',
    help='width, height'
)
parser.add_argument(
    '--scheduler',
    action='store_true',
)
parser.add_argument(
    '--scheduler-epochs',
    dest='scheduler_epochs',
    default=[50],
    nargs='+',
    type=int
)
args = parser.parse_args()
logger.info(f"Arguments: {args}")

if __name__ == '__main__':
    # Create output directories
    out_dir = os.path.join('outputs_large')
    out_dir_valid_preds = os.path.join(out_dir, 'valid_preds')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_valid_preds, exist_ok=True)

    # Check device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    model, processor = load_model(num_classes=len(ALL_CLASSES))
    model = model.to(device)
    logger.info(f"Model loaded: {model}")
    
    # Total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{total_params:,} total parameters.")
    logger.info(f"{total_trainable_params:,} training parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Get dataset
    train_images, train_masks, valid_images, valid_masks = get_images(
        root_path='/WD/심신개/new_train2'    
    )

    train_dataset, valid_dataset = get_dataset(
        train_images, 
        train_masks,
        valid_images,
        valid_masks,
        ALL_CLASSES,
        ALL_CLASSES,
        LABEL_COLORS_LIST,
        img_size=args.imgsz,
        feature_extractor=processor
    )

    train_dataloader, valid_dataloader = get_data_loaders(
        train_dataset, 
        valid_dataset,
        args.batch,
        processor
    )

    # Initialize SaveBestModel classes
    save_best_model = SaveBestModel()
    save_best_iou = SaveBestModelIOU()

    # LR Scheduler
    scheduler = MultiStepLR(
        optimizer, milestones=args.scheduler_epochs, gamma=0.1, verbose=True
    )

    train_loss, train_miou = [], []
    valid_loss, valid_miou = [], []
    
    metric = evaluate_func.load("mean_iou")

    # Training loop
    for epoch in range(args.epochs):
        logger.info(f"EPOCH: {epoch + 1}")
        train_epoch_loss, train_epoch_miou = train(
            model,
            train_dataloader,
            device,
            optimizer,
            ALL_CLASSES,
            processor,
            metric
        )
        valid_epoch_loss, valid_epoch_miou = validate(
            model,
            valid_dataloader,
            device,
            ALL_CLASSES,
            LABEL_COLORS_LIST,
            epoch,
            save_dir=out_dir_valid_preds,
            processor=processor,
            metric=metric
        )

        # Logging the results for each epoch
        logger.info(f"Train Epoch Loss: {train_epoch_loss:.4f}, Train Epoch mIOU: {train_epoch_miou:.4f}")
        logger.info(f"Valid Epoch Loss: {valid_epoch_loss:.4f}, Valid Epoch mIOU: {valid_epoch_miou:.4f}")

        train_loss.append(train_epoch_loss)
        train_miou.append(train_epoch_miou)
        valid_loss.append(valid_epoch_loss)
        valid_miou.append(valid_epoch_miou)

        save_best_model(
            valid_epoch_loss, epoch, model, out_dir, name='model_loss'
        )
        save_best_iou(
            valid_epoch_miou, epoch, model, out_dir, name='model_iou'
        )

        if args.scheduler:
            scheduler.step()

        logger.info('-' * 50)

    # Save the loss and accuracy plots
    save_plots(
        train_loss, valid_loss,
        train_miou, valid_miou, 
        out_dir
    )

    # Save the final model
    save_model(model, out_dir, name='final_model')
    logger.info('TRAINING COMPLETE')
