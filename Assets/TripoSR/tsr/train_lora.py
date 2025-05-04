import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from omegaconf import OmegaConf
from lpips import LPIPS
from .system import TSR
from .utils import ImagePreprocessor
import trimesh
import pytorch3d
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, SoftPhongShader,
    PerspectiveCameras, PointLights
)
import logging
from pathlib import Path
import wandb
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TripoSRDataset(Dataset):
    def __init__(self, data_dir, image_size=512):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.image_paths = list(self.data_dir.glob("*.png"))
        self.image_processor = ImagePreprocessor()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No PNG images found in {data_dir}")
        logger.info(f"Found {len(self.image_paths)} images in {data_dir}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.image_processor(image, self.image_size)
        return {'image': image, 'path': str(image_path)}

class TripoSRTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accelerator = Accelerator(mixed_precision='bf16')
        
        # 初始化模型
        self.model = self._init_model()
        
        # 初始化数据集和数据加载器
        self.train_dataset = TripoSRDataset(
            data_dir=config.data.train_dir,
            image_size=config.data.image_size
        )
        self.val_dataset = TripoSRDataset(
            data_dir=config.data.val_dir,
            image_size=config.data.image_size
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=4
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # 初始化优化器和调度器
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.training.lr
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.max_steps,
            eta_min=config.training.lr / 10
        )
        
        # 初始化损失函数
        self.mse_loss = torch.nn.MSELoss()
        self.lpips_loss = LPIPS(net='vgg').to(self.device)
        
        # 准备模型和数据加载器
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )
        
        # 初始化wandb
        if self.accelerator.is_main_process:
            wandb.init(
                project=config.wandb.project_name,
                config=OmegaConf.to_container(config, resolve=True)
            )
    
    def _init_model(self):
        model = TSR.from_pretrained(
            pretrained_model_name_or_path=self.config.model.pretrained_path,
            config_name=self.config.model.config_name,
            weight_name=self.config.model.weight_name
        )
        
        # 应用LoRA
        lora_config = LoraConfig(
            r=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            target_modules=["self_attention.query", "self_attention.value",
                          "cross_attention.query", "cross_attention.value"],
            lora_dropout=0.1
        )
        model = get_peft_model(model, lora_config)
        
        # 冻结非LoRA参数
        for name, param in model.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
        
        return model
    
    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        images = batch['image'].to(self.device)
        
        # 前向传播
        scene_codes = self.model.forward(images, device=self.device)
        rendered_views = self.model.render(
            scene_codes,
            n_views=4,
            elevation_deg=0.0,
            camera_distance=1.9,
            fovy_deg=40.0,
            height=self.config.data.image_size,
            width=self.config.data.image_size
        )
        
        # 计算损失
        mse = self.mse_loss(rendered_views, images)
        lpips = self.lpips_loss(rendered_views, images).mean()
        loss = mse + self.config.training.lambda_lpips * lpips
        
        # 反向传播
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'mse': mse.item(),
            'lpips': lpips.item()
        }
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_lpips = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                scene_codes = self.model.forward(images, device=self.device)
                rendered_views = self.model.render(
                    scene_codes,
                    n_views=4,
                    elevation_deg=0.0,
                    camera_distance=1.9,
                    fovy_deg=40.0,
                    height=self.config.data.image_size,
                    width=self.config.data.image_size
                )
                
                mse = self.mse_loss(rendered_views, images)
                lpips = self.lpips_loss(rendered_views, images).mean()
                loss = mse + self.config.training.lambda_lpips * lpips
                
                total_loss += loss.item()
                total_mse += mse.item()
                total_lpips += lpips.item()
        
        num_batches = len(self.val_loader)
        return {
            'val_loss': total_loss / num_batches,
            'val_mse': total_mse / num_batches,
            'val_lpips': total_lpips / num_batches
        }
    
    def train(self):
        best_val_loss = float('inf')
        patience = self.config.training.patience
        patience_counter = 0
        
        for epoch in range(self.config.training.num_epochs):
            # 训练一个epoch
            epoch_losses = []
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
            for batch in progress_bar:
                losses = self.train_step(batch)
                epoch_losses.append(losses)
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f"{losses['loss']:.4f}",
                    'mse': f"{losses['mse']:.4f}",
                    'lpips': f"{losses['lpips']:.4f}"
                })
            
            # 计算平均训练损失
            avg_losses = {
                k: np.mean([l[k] for l in epoch_losses])
                for k in epoch_losses[0].keys()
            }
            
            # 验证
            val_metrics = self.validate()
            
            # 记录指标
            if self.accelerator.is_main_process:
                metrics = {**avg_losses, **val_metrics}
                wandb.log(metrics)
                
                # 保存检查点
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    patience_counter = 0
                    self.save_checkpoint(epoch, val_metrics)
                else:
                    patience_counter += 1
                
                # 早停
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
    
    def save_checkpoint(self, epoch, metrics):
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

def main():
    # 默认配置
    config = OmegaConf.create({
        'model': {
            'pretrained_path': 'stabilityai/TripoSR',
            'config_name': 'config.yaml',
            'weight_name': 'model.pth'
        },
        'lora': {
            'rank': 16,
            'alpha': 32
        },
        'data': {
            'train_dir': './data/train',
            'val_dir': './data/val',
            'image_size': 512
        },
        'training': {
            'batch_size': 4,
            'lr': 4e-4,
            'max_steps': 10000,
            'num_epochs': 100,
            'patience': 10,
            'lambda_lpips': 2.0,
            'checkpoint_dir': './checkpoints'
        },
        'wandb': {
            'project_name': 'triposr-lora'
        }
    })
    
    # 加载自定义配置（如果存在）
    if os.path.exists('config.yaml'):
        custom_config = OmegaConf.load('config.yaml')
        config = OmegaConf.merge(config, custom_config)
    
    # 创建训练器并开始训练
    trainer = TripoSRTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main() 