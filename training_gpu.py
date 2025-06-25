import os
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class KoreanEquityDataset(Dataset):
    def __init__(self, intervals: int, years: List[int], base_dir: str = 'Images') -> None:
        self.intervals = intervals
        self.base_dir = os.path.join(os.path.dirname(
            __file__), base_dir, str(self.intervals))

        metadata_path = os.path.join(
            self.base_dir, f'charts_{self.intervals}d_metadata.feather')
        images_path = os.path.join(
            self.base_dir, f'images_{self.intervals}d.npy')

        if not os.path.exists(metadata_path) or not os.path.exists(images_path):
            raise FileNotFoundError(
                f"Data files not found in {self.base_dir}. Please run convert_data.py first.")

        all_metadata = pd.read_feather(metadata_path)
        all_metadata['start_date'] = pd.to_datetime(all_metadata['start_date'])

        self.metadata = all_metadata[all_metadata['start_date'].dt.year.isin(
            years)].reset_index()

        print("Loading image data via memory-mapping...")
        self.images = np.load(images_path, mmap_mode='r')
        print("Image data memory-mapped successfully.")

        self.image_height = self.images.shape[1]
        self.image_width = self.images.shape[2]

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict:
        row = self.metadata.iloc[idx]

        original_image_idx = row.image_idx
        img_array = self.images[original_image_idx]

        img_tensor = torch.from_numpy(
            img_array.astype(np.float32)).unsqueeze(0)

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
        bn_loc: str = "bn_bf_relu",
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
            bn_loc,
            conv_layer_chanls
        )
        fc_size = self._get_conv_layers_flatten_size()

        self.fc = nn.Linear(
            fc_size, 1) if regression_label is not None else nn.Linear(fc_size, 2)

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
        bn_loc: str,
    ) -> nn.Sequential:

        layers = []

        if bn_loc == "bn_bf_relu":
            layers.extend([
                nn.Conv2d(in_chanl, out_chanl, filter_size,
                          stride=stride, padding=padding, dilation=dilation),
                nn.BatchNorm2d(out_chanl),
                nn.LeakyReLU(negative_slope=0.01) if lrelu else nn.ReLU(),
            ])
        elif bn_loc == "bn_af_relu":
            layers.extend([
                nn.Conv2d(in_chanl, out_chanl, filter_size,
                          stride=stride, padding=padding, dilation=dilation),
                nn.LeakyReLU(negative_slope=0.01) if lrelu else nn.ReLU(),
                nn.BatchNorm2d(out_chanl),
            ])
        else:
            layers.extend([
                nn.Conv2d(in_chanl, out_chanl, filter_size,
                          stride=stride, padding=padding, dilation=dilation),
                nn.LeakyReLU(negative_slope=0.01) if lrelu else nn.ReLU(),
            ])

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
        bn_loc: str,
        conv_layer_chanls: Optional[List[int]],
    ) -> nn.Sequential:

        if conv_layer_chanls is None:
            conv_layer_chanls = [inplanes * (2**i)
                                 for i in range(layer_number)]

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
                bn_loc=bn_loc if batch_norm else "none",
            )
            layers.append(layer)
            prev_chanl = out_chanl

        layers.append(Flatten())
        layers.append(nn.Dropout(p=drop_prob))
        return nn.Sequential(*layers)

    def _get_conv_layers_flatten_size(self) -> int:
        dummy_input = torch.rand(
            (1, 1, self.input_size[0], self.input_size[1]))
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

        self.device = self._get_optimal_device()
        print(f"Using device: {self.device}")

        if self.device.type == 'mps':
            print("M1 GPU acceleration enabled via Metal Performance Shaders")
            torch.backends.mps.enabled = True

        self.exp_name = f"korea_cnn_{ws}d{pw}p_{config['mode']}_gpu"
        self.model_dir = os.path.join(
            os.path.dirname(__file__), 'models', self.exp_name)
        os.makedirs(self.model_dir, exist_ok=True)

    def _get_optimal_device(self) -> torch.device:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("Warning: No GPU acceleration available, falling back to CPU")
            return torch.device("cpu")

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

        print(
            f"Train size: {len(train_dataset)} ({train_ratio*100:.0f}%), Validation size: {len(val_dataset)} ({(1-train_ratio)*100:.0f}%)")

        num_workers = min(8, os.cpu_count() or 1)
        print(
            f"Using {num_workers} workers for data loading (optimized for M1)")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.type != 'cpu',
            persistent_workers=True if num_workers > 0 else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type != 'cpu',
            persistent_workers=True if num_workers > 0 else False
        )

        return {"train": train_loader, "validate": val_loader}

    def train_empirical_ensem_model(self, dataloaders_dict: Dict[str, DataLoader]) -> None:
        for model_num in range(self.config['ensem_size']):
            print(
                f"\n--- Training Ensemble Member {model_num + 1}/{self.config['ensem_size']} ---")
            model_save_path = os.path.join(
                self.model_dir, f"checkpoint{model_num}.pth.tar")
            self.train_single_model(dataloaders_dict, model_save_path)

    def train_single_model(self, dataloaders_dict: Dict[str, DataLoader], model_save_path: str) -> None:
        ds = dataloaders_dict['train'].dataset

        paddings = [(int(fs[0] / 2), int(fs[1] / 2))
                    for fs in self.config['filter_sizes']]

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

        optimizer = optim.AdamW(
            model.parameters(), lr=self.config['lr'], weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(self.config['max_epoch']):
            for phase in ['train', 'validate']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                loader = dataloaders_dict[phase]
                for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{self.config['max_epoch']} - {phase}"):
                    inputs = batch['image'].to(self.device, non_blocking=True)
                    labels = batch['label'].to(self.device, non_blocking=True)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(loader.dataset)
                epoch_acc = running_corrects.double() / len(loader.dataset)
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'validate':
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        best_model_state = model.state_dict()
                        epochs_no_improve = 0
                        torch.save({'epoch': epoch, 'model_state_dict': best_model_state,
                                   'loss': best_val_loss}, model_save_path)
                        print(f"New best model saved to {model_save_path}")
                    else:
                        epochs_no_improve += 1

            if epochs_no_improve >= 3:
                print("Early stopping triggered.")
                break

            if self.device.type == 'mps':
                torch.mps.empty_cache()


def main():
    RUN_MODE = 'TEST'

    MODE_CONFIGS = {
        'TEST': {
            'mode': 'test',
            'train_years': list(range(2000, 2011)),
            'ensem_size': 1,
            'batch_size': 128,
            'max_epoch': 10,
            'lr': 2e-4,
            'drop_prob': 0.3,
            'conv_channels': [32, 64, 128],
        },
        'PRODUCTION': {
            'mode': 'production',
            'train_years': list(range(2000, 2011)),
            'ensem_size': 5,
            'batch_size': 256,
            'max_epoch': 50,
            'lr': 1e-4,
            'drop_prob': 0.5,
            'conv_channels': [32, 64, 128, 256],
        }
    }

    WINDOW_CONFIGS = {
        5: {
            'pw': 5,
            'filter_sizes': {
                'TEST': [(3, 2), (3, 2), (3, 2)],
                'PRODUCTION': [(3, 2), (3, 2), (3, 2), (3, 2)]
            }
        },
        20: {
            'pw': 20,
            'filter_sizes': {
                'TEST': [(5, 3), (3, 3), (3, 3)],
                'PRODUCTION': [(5, 3), (3, 3), (3, 3), (3, 3)]
            }
        },
        60: {
            'pw': 60,
            'filter_sizes': {
                'TEST': [(5, 3), (5, 3), (3, 3)],
                'PRODUCTION': [(5, 3), (5, 3), (3, 3), (3, 3)]
            }
        }
    }

    for ws, window_config in WINDOW_CONFIGS.items():
        pw = window_config['pw']
        print(f"\n{'='*25} TRAINING MODEL: {ws}d{pw}p {'='*25}")
        print(f"--- Running in {RUN_MODE} MODE (MacBook M1 Optimized) ---")

        config = MODE_CONFIGS[RUN_MODE].copy()
        config['filter_sizes'] = window_config['filter_sizes'][RUN_MODE]

        trainer = Trainer(ws=ws, pw=pw, config=config)

        dataloaders = trainer.get_dataloaders(
            train_years=config['train_years'])
        trainer.train_empirical_ensem_model(dataloaders)


def main_60d():
    """
    M1 optimized version for 60d data training with fewer epochs.
    """
    RUN_MODE = 'TEST'

    MODE_CONFIGS = {
        'TEST': {
            'mode': 'test',
            'train_years': list(range(2000, 2011)),
            'ensem_size': 1,
            'batch_size': 128,
            'max_epoch': 5,
            'lr': 2e-4,
            'drop_prob': 0.3,
            'conv_channels': [32, 64, 128],
        }
    }

    ws = 60
    pw = 60
    filter_sizes = [(5, 3), (5, 3), (3, 3)]

    print(f"\n{'='*25} TRAINING MODEL: {ws}d{pw}p {'='*25}")
    print(f"--- Running in {RUN_MODE} MODE (MacBook M1 Optimized) ---")

    config = MODE_CONFIGS[RUN_MODE].copy()
    config['filter_sizes'] = filter_sizes

    trainer = Trainer(ws=ws, pw=pw, config=config)
    dataloaders = trainer.get_dataloaders(train_years=config['train_years'])
    trainer.train_empirical_ensem_model(dataloaders)


if __name__ == "__main__":
    main_60d()
