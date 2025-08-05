import os
import shutil
import torch
from tqdm import tqdm

from data_loader import load_data
from models import *

def organize_files(path):
    if not os.path.exists(path):
        return
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if files:
        new_folder = os.path.join(path, "organized_files")
        os.makedirs(new_folder, exist_ok=True)
        for file in files:
            shutil.move(os.path.join(path, file), os.path.join(new_folder, file))
        print(f"\u5df2\u5c06 {len(files)} \u4e2a\u6587\u4ef6\u79fb\u52a8\u5230 {new_folder}")

def data_load_split(args):
    TrainDataset, TrainDataloader = load_data(args, flag='train')
    TestDataset, TestDataloader = load_data(args, flag='test')
    print('train len: ', len(TrainDataset))
    print('test len: ', len(TestDataset))
    return TrainDataset, TrainDataloader, TestDataset, TestDataloader

def train(model, criterion, optimizer, scheduler, writer, TrainDataloader, TestDataloader, args, device):
    model.train()
    for epoch in range(args['epoch_num']):
        total_loss = 0.0
        for step, (macro_x, merged_x, conf_x, y) in enumerate(tqdm(TrainDataloader, desc=f'epoch {epoch+1}')):
            macro_x = macro_x.to(device).float()
            merged_x = merged_x.to(device).float()
            conf_x = conf_x.to(device).float()
            y = y.to(device).float()

            optimizer.zero_grad()
            outputs = model(merged_x, conf_x, macro_x)  # inputs: financial, article, macro

            loss = criterion(outputs, y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            if args['lr_strategy'] == 'cosine':
                scheduler.step()
            writer.add_scalar('Loss/train_step', loss.item(), epoch * len(TrainDataloader) + step)

        if args['lr_strategy'] == 'expo':
            scheduler.step()

        val(model, criterion, writer, TrainDataloader, epoch, device, 'train')
        val(model, criterion, writer, TestDataloader, epoch, device, 'val')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }, f"./res/{args['log_sub_dir']}/epoch_{epoch+1}.pth")

        avg_loss = total_loss / len(TrainDataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

def val(model, criterion, writer, ValDataloader, epoch, device, tag='val'):
    model.eval()
    val_loss = 0.0
    mae_sum = None
    direction_correct = None
    total_samples = 0

    with torch.no_grad():
        for macro_x, merged_x, conf_x, y in ValDataloader:
            macro_x = macro_x.to(device).float()
            merged_x = merged_x.to(device).float()
            conf_x = conf_x.to(device).float()
            y = y.to(device).float()

            outputs = model(merged_x, conf_x, macro_x)
            loss = criterion(outputs, y)
            val_loss += loss.item()

            abs_error = torch.abs(outputs - y)
            correct_sign = (torch.sign(outputs) == torch.sign(y)).float()

            if mae_sum is None:
                mae_sum = abs_error.sum(dim=0)
                direction_correct = correct_sign.sum(dim=0)
            else:
                mae_sum += abs_error.sum(dim=0)
                direction_correct += correct_sign.sum(dim=0)
            total_samples += y.size(0)

    val_loss /= len(ValDataloader)
    mae = (mae_sum / total_samples).cpu().numpy()
    direction_acc = (direction_correct / total_samples).cpu().numpy()

    print(f"Val loss of {tag}: {val_loss:.4f}")
    print(f"MAE per dim: {mae}")
    print(f"Direction acc per dim: {direction_acc}")

    writer.add_scalar(f'Loss/{tag}', val_loss, epoch)
    for i in range(len(mae)):
        writer.add_scalar(f'MAE/{tag}_dim_{i}', mae[i], epoch)
        writer.add_scalar(f'DirectionAcc_{tag}/dim_{i}', direction_acc[i], epoch)

def test(model, TestDataloader, writer, device):
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for macro_x, merged_x, conf_x, y in TestDataloader:
            macro_x = macro_x.to(device).float()
            merged_x = merged_x.to(device).float()
            conf_x = conf_x.to(device).float()
            y = y.to(device).float()

            outputs = model(merged_x, conf_x, macro_x)
            loss = criterion(outputs, y)
            total_loss += loss.item()

    avg_loss = total_loss / len(TestDataloader)
    writer.add_scalar('Loss/test', avg_loss, 0)
    print(f"Test MSE Loss: {avg_loss:.4f}")


if __name__ == '__main__':
    with open('args.json') as f:
        import json
        args = json.load(f)
        print(args)
    TrainDataset, TrainDataloader, TestDataset, TestDataloader = data_load_split(args)
    for step, (macro_x, merged_x, conf_x, y) in enumerate(tqdm(TrainDataloader, desc=f'train')):
        pass
