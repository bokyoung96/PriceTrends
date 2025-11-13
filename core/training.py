import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
IMAGES_ROOT = ROOT / "Images"
MODELS_ROOT = ROOT / "models"

from core.params import CNNParams
from core.device import DeviceSelector

logger = logging.getLogger(__name__)


class KoreanEquityDataset(Dataset):
    def __init__(self, intervals: int, years: List[int], base_dir: Optional[str] = None) -> None:
        self.intervals = intervals
        base_path = Path(base_dir) if base_dir is not None else IMAGES_ROOT
        self.base_path = base_path / str(self.intervals)
        
        metadata_path = self.base_path / f'charts_{self.intervals}d_metadata.feather'
        images_path = self.base_path / f'images_{self.intervals}d.npy'

        if not metadata_path.exists() or not images_path.exists():
            raise FileNotFoundError(f"Data files not found in {self.base_path}. Please run convert_data.py first.")

        all_metadata = pd.read_feather(metadata_path)
        all_metadata['start_date'] = pd.to_datetime(all_metadata['start_date'])
        
        self.metadata = all_metadata[all_metadata['start_date'].dt.year.isin(years)].reset_index()
        
        logger.info("Loading image data via memory-mapping...")
        self.images = np.load(images_path, mmap_mode='r')
        if self.images.dtype != np.float32:
            logging.warning(
                "Images dtype is %s; batches will be cast to float32 on the fly.",
                self.images.dtype,
            )
        logger.info("Image data memory-mapped successfully.")
        
        self.image_height = self.images.shape[1]
        self.image_width = self.images.shape[2]

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict:
        row = self.metadata.iloc[idx]
        
        original_image_idx = row.image_idx
        img_array = self.images[original_image_idx]
        if img_array.dtype != np.float32:
            img_array = img_array.astype(np.float32, copy=False)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        label = torch.tensor(row.label, dtype=torch.long)

        return {
            'image': img_tensor,
            'label': label,
            'StockID': row.ticker,
            'ending_date': row.end_date,
        }

def init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)

class CNNModel(nn.Module):
    def __init__(
        self,
        layer_number: int,
        input_size: Tuple[int, int],
        inplanes: int,
        drop_prob: float,
        filter_size_list: List[Tuple[int, int]],
        stride_list: List[Tuple[int, int]],
        padding_list: List[Tuple[int, int]],
        dilation_list: List[Tuple[int, int]],
        max_pooling_list: List[Tuple[int, int]],
        batch_norm: bool = True,
        xavier: bool = True,
        lrelu: bool = True,
        conv_layer_chanls: Optional[List[int]] = None,
        regression_label: Optional[str] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.conv_layers = self._init_conv_layers(
            layer_number,
            inplanes,
            drop_prob,
            filter_size_list,
            stride_list,
            padding_list,
            dilation_list,
            max_pooling_list,
            batch_norm,
            lrelu,
            conv_layer_chanls
        )
        fc_size = self._get_conv_layers_flatten_size()
        
        self.fc = nn.Linear(fc_size, 1) if regression_label is not None else nn.Linear(fc_size, 2)
        
        if xavier:
            self.apply(init_weights)

    @staticmethod
    def _conv_layer(
        in_chanl: int,
        out_chanl: int,
        filter_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        max_pooling: Tuple[int, int],
        lrelu: bool,
        batch_norm: bool,
    ) -> nn.Sequential:
        
        layers: List[nn.Module] = [
            nn.Conv2d(in_chanl, out_chanl, filter_size, stride=stride, padding=padding, dilation=dilation)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_chanl))
        layers.append(nn.LeakyReLU(negative_slope=0.01) if lrelu else nn.ReLU())
        
        if max_pooling and max_pooling != (1, 1):
            layers.append(nn.MaxPool2d(kernel_size=max_pooling))
            
        return nn.Sequential(*layers)

    def _init_conv_layers(
        self,
        layer_number: int,
        inplanes: int,
        drop_prob: float,
        filter_size_list: List[Tuple[int, int]],
        stride_list: List[Tuple[int, int]],
        padding_list: List[Tuple[int, int]],
        dilation_list: List[Tuple[int, int]],
        max_pooling_list: List[Tuple[int, int]],
        batch_norm: bool,
        lrelu: bool,
        conv_layer_chanls: Optional[List[int]],
    ) -> nn.Sequential:
        
        if conv_layer_chanls is None:
            conv_layer_chanls = [inplanes * (2**i) for i in range(layer_number)]

        layers = []
        prev_chanl = 1
        for i, out_chanl in enumerate(conv_layer_chanls):
            layer = self._conv_layer(
                prev_chanl,
                out_chanl,
                filter_size=filter_size_list[i],
                stride=stride_list[i],
                padding=padding_list[i],
                dilation=dilation_list[i],
                max_pooling=max_pooling_list[i],
                lrelu=lrelu,
                batch_norm=batch_norm,
            )
            layers.append(layer)
            prev_chanl = out_chanl
            
        layers.append(Flatten())
        layers.append(nn.Dropout(p=drop_prob))
        return nn.Sequential(*layers)

    def _get_conv_layers_flatten_size(self) -> int:
        dummy_input = torch.rand((1, 1, self.input_size[0], self.input_size[1]))
        with torch.no_grad():
            x = self.conv_layers(dummy_input)
        return x.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

class Trainer:
    def __init__(self, ws: int, pw: int, config: Dict) -> None:
        self.ws = ws
        self.pw = pw
        self.config = config
        
        selector = DeviceSelector()
        self.device = selector.resolve()
        logger.info(selector.summary("Trainer"))

        self.exp_name = f"korea_cnn_{ws}d{pw}p_{config['mode']}"
        self.model_dir = MODELS_ROOT / self.exp_name
        os.makedirs(self.model_dir, exist_ok=True)

    def get_dataloaders(self, train_years: List[int], train_ratio: float = 0.7) -> Dict[str, DataLoader]:
        full_dataset = KoreanEquityDataset(self.ws, train_years)
        
        train_indices, val_indices = train_test_split(
            range(len(full_dataset)), 
            test_size=1.0 - train_ratio, 
            random_state=42,
            stratify=full_dataset.metadata['label']
        )
        
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        
        logger.info(
            "Train size: %d (%.0f%%), Validation size: %d (%.0f%%)",
            len(train_dataset),
            train_ratio * 100,
            len(val_dataset),
            (1 - train_ratio) * 100,
        )
        
        num_workers = self.config.get('num_workers', 0)
        spawn_like = os.name == "nt"
        if not spawn_like:
            try:
                start_method = mp.get_start_method(allow_none=True)
            except RuntimeError:
                start_method = None
            spawn_like = start_method in (None, "spawn", "forkserver")
        if num_workers > 0 and spawn_like:
            logger.warning(
                "Detected spawn-based multiprocessing; forcing num_workers=0 because numpy.memmap datasets cannot be pickled safely."
            )
            num_workers = 0
        pin_memory = self.device.type == 'cuda'
        dataloader_kwargs = {
            "batch_size": self.config['batch_size'],
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = True
        
        train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
        
        return {"train": train_loader, "validate": val_loader}

    def train_empirical_ensem_model(self, dataloaders_dict: Dict[str, DataLoader]) -> None:
        for model_num in range(self.config['ensem_size']):
            logger.info(
                "\n--- Training Ensemble Member %d/%d ---",
                model_num + 1,
                self.config['ensem_size'],
            )
            model_save_path = self.model_dir / f"checkpoint{model_num}.pth.tar"
            self.train_single_model(dataloaders_dict, model_save_path)

    def train_single_model(self, dataloaders_dict: Dict[str, DataLoader], model_save_path: str) -> None:
        ds = dataloaders_dict['train'].dataset
        
        paddings = [(int(fs[0] / 2), int(fs[1] / 2)) for fs in self.config['filter_sizes']]
        
        original_ds = ds.dataset if hasattr(ds, 'dataset') else ds
        
        model = CNNModel(
            layer_number=len(self.config['conv_channels']),
            input_size=(original_ds.image_height, original_ds.image_width),
            inplanes=self.config['conv_channels'][0],
            conv_layer_chanls=self.config['conv_channels'],
            drop_prob=self.config['drop_prob'],
            filter_size_list=self.config['filter_sizes'],
            stride_list=[(1, 1)] * len(self.config['conv_channels']),
            padding_list=paddings,
            dilation_list=[(1, 1)] * len(self.config['conv_channels']),
            max_pooling_list=[(2, 1)] * len(self.config['conv_channels']),
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.config['lr'])
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(self.config['max_epoch']):
            for phase in ['train', 'validate']:
                is_train = phase == 'train'
                model.train() if is_train else model.eval()

                running_loss = 0.0
                running_corrects = 0
                
                loader = dataloaders_dict[phase]
                grad_context = torch.enable_grad if is_train else torch.no_grad
                with grad_context():
                    for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{self.config['max_epoch']} - {phase}"):
                        inputs = batch['image'].to(self.device, non_blocking=True)
                        labels = batch['label'].to(self.device, non_blocking=True)

                        if is_train:
                            optimizer.zero_grad(set_to_none=True)

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if is_train:
                            loss.backward()
                            optimizer.step()
                        
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(loader.dataset)
                epoch_acc = running_corrects.double() / len(loader.dataset)
                logger.info('%s Loss: %.4f Acc: %.4f', phase, epoch_loss, epoch_acc)

                if not is_train:
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        best_model_state = model.state_dict()
                        epochs_no_improve = 0
                        torch.save({'epoch': epoch, 'model_state_dict': best_model_state, 'loss': best_val_loss}, model_save_path)
                        logger.info("New best model saved to %s", model_save_path)
                    else:
                        epochs_no_improve += 1
            
            if epochs_no_improve >= 3:
                logger.info("Early stopping triggered.")
            break


def main(windows: Optional[List[int]] = None):
    RUN_MODE = 'TEST'
    params = CNNParams()
    default_windows = params.window_sizes
    target_windows = windows if windows is not None else default_windows

    for ws in target_windows:
        if ws not in default_windows:
            logger.warning("Skipping window %d: not defined in config.json", ws)
            continue

        pw = params.config['window_configs'][str(ws)]['pw']
        logger.info("\n%s TRAINING MODEL: %dd%dp %s", '=' * 25, ws, pw, '=' * 25)
        logger.info("--- Running in %s MODE ---", RUN_MODE)

        config = params.get_config(RUN_MODE, ws)
        trainer = Trainer(ws=ws, pw=config['pw'], config=config)
        
        dataloaders = trainer.get_dataloaders(train_years=config['train_years'])
        trainer.train_empirical_ensem_model(dataloaders)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    WINDOWS_TO_TRAIN = [5, 20, 60]
    main(windows=WINDOWS_TO_TRAIN)
