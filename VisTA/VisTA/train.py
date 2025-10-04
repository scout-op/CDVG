import os

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist

from utils.logger import get_logger

import argparse

from dataset.DataNew import DatasetNew
from model.VisTA import VisTA
from hrpg.qwen_reasoner import QwenVLReasoner
from hrpg.config import HRPGConfig
from hrpg.prior_encoder import encode_prior_to_prompts
import torchvision.transforms as T

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

batch_size = 16
workers = 8
lr=1e-4
max_epoch=50

TITLE = 'VisTA'

# Directories and logger will be set up inside main(args) based on CLI.
scaler = torch.GradScaler()

# Map type string to index (must align with VisTA.TYPE_NAMES)
TYPE2IDX = {
    'change_ratio': 0,
    'change_ratio_types': 0,  # alias seen in evaluation
    'change_or_not': 1,
    'change_to_what': 2,
    'increase_or_not': 3,
    'decrease_or_not': 4,
    'smallest_change': 5,
    'largest_change': 6,
    'change_from_what': 7,
}


def confuse_matrix(pre, label, n_class):
    pre = pre.cpu().numpy()
    label = label.cpu().numpy()
    cm = np.bincount(label * n_class + pre, minlength=n_class ** 2).reshape(n_class, n_class)
    return cm


def ddp_setup():
    # Rely on torchrun to provide MASTER_ADDR/MASTER_PORT/LOCAL_RANK via environment variables.
    # This avoids hard-coding ports and prevents socket timeouts due to conflicts.
    init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))


def _to_pil_batch(x):
    # x: (B,3,H,W) tensor in [0,1] or similar
    to_pil = T.ToPILImage()
    x = x.float().clamp(0, 1).cpu()
    return [to_pil(xi) for xi in x]


def build_hrpg_points(reasoner: QwenVLReasoner,
                      imgs1, imgs2, questions,
                      input_image_size=(512, 512),
                      k_points_per_box: int = 1):
    """Return batched (points, labels) in PromptEncoder coordinate space.
    - Always returns a fixed K across batch by padding with label=-1.
    - Boxes from VLM are converted to centers as points for stability.
    """
    B, _, H, W = imgs2.shape
    pil1 = _to_pil_batch(imgs1)
    pil2 = _to_pil_batch(imgs2)

    # Collect per-sample points
    pts_list = []
    max_k = 0
    for i in range(B):
        q = questions[i] if isinstance(questions, (list, tuple)) else questions
        res = reasoner.predict(pil1[i], pil2[i], q)
        prior = res.get('prior', None)
        # Encode to points/boxes in prompt space; then scale from src (H,W) to (input_h,input_w)
        p_single, b_single = encode_prior_to_prompts(prior, prompt_image_size=input_image_size, k_points_per_box=k_points_per_box)
        # encode_prior_to_prompts assumes coords already in prompt space; here we have boxes in image space
        # To be robust, we will rescale manually from (H,W)->input_image_size if points exist
        if p_single is not None:
            coords, labels = p_single
            # We don't know the original coord system of VLM outputs; assume current PIL size (W,H)
            sy = float(input_image_size[0]) / float(max(1, H))
            sx = float(input_image_size[1]) / float(max(1, W))
            coords[..., 0] = coords[..., 0] * sx
            coords[..., 1] = coords[..., 1] * sy
        # Prefer points; if None, create empty placeholders
        if p_single is None:
            coords = torch.zeros((1, 0, 2), dtype=torch.float32)
            labels = torch.zeros((1, 0), dtype=torch.long)
        pts_list.append((coords.squeeze(0), labels.squeeze(0)))
        if coords.numel() // 2 > max_k:
            max_k = coords.shape[0] if coords.dim() == 2 else 0

    # Pad to fixed K with label -1 (not-a-point)
    K = max_k if max_k > 0 else 1
    batched_coords = torch.zeros((B, K, 2), dtype=torch.float32)
    batched_labels = -torch.ones((B, K), dtype=torch.long)
    for i, (coords, labels) in enumerate(pts_list):
        if coords.numel() == 0:
            continue
        k = min(K, coords.shape[0])
        batched_coords[i, :k] = coords[:k]
        # set positive labels=1 for provided points; rest remain -1
        batched_labels[i, :k] = 1
    return (batched_coords, batched_labels)


def main(args):
    ddp_setup()

    # Configure save/log directories
    ckp_savepath = args.ckp_dir if getattr(args, 'ckp_dir', '') else ('./ckps/' + TITLE)
    log_savepath = args.log_dir if getattr(args, 'log_dir', '') else './logs/'
    if not os.path.exists(ckp_savepath):
        os.makedirs(ckp_savepath, exist_ok=True)
    if not os.path.exists(log_savepath):
        os.makedirs(log_savepath, exist_ok=True)
    logger = get_logger(os.path.join(log_savepath, TITLE + '.log'))

    train_root = './train/'
    val_root='./val/'


    train_dataset = DatasetNew(train_root+'json/Train.json', train_root+'image/', train_root+'mask/')
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True,
                              sampler=train_sampler)

    val_dataset = DatasetNew(val_root+'json/Val.json', val_root+'image/', val_root+'mask/')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True,
                            sampler=DistributedSampler(val_dataset))

    criterion = nn.CrossEntropyLoss()

    gpu_id = int(os.environ.get('LOCAL_RANK', 0))

    model=VisTA()

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model.to(gpu_id), device_ids=[gpu_id], find_unused_parameters=True)

    # Override tunable hyperparameters from CLI
    try:
        mm = model.module
        mm.sample_ratio = float(getattr(args, 'sample_ratio', mm.sample_ratio))
        mm.pixel_hard_ratio = float(getattr(args, 'pixel_hard_ratio', mm.pixel_hard_ratio))
        mm.contrast_weight = float(getattr(args, 'contrast_weight', mm.contrast_weight))
        mm.temp_reg_weight = float(getattr(args, 'temp_reg_weight', mm.temp_reg_weight))
        mm.type_weight = float(getattr(args, 'type_weight', mm.type_weight))
        mm.consistency_weight = float(getattr(args, 'consistency_weight', mm.consistency_weight))
        mm.contrast_tau = float(getattr(args, 'contrast_tau', mm.contrast_tau))
        mm.enable_contrast = bool(getattr(args, 'enable_contrast', 1))
        mm.enable_uncertainty = bool(getattr(args, 'enable_uncertainty', 1))
        mm.enable_hard_mining = bool(getattr(args, 'enable_hard_mining', 1))
    except Exception as _e:
        if gpu_id == 0:
            logger.warning(f"Failed to override model hyperparameters: {_e}")


    # HRPG (optional): prepare reasoner per rank if enabled
    hrpg_reasoner = None
    if int(getattr(args, 'use_hrpg', 0)) == 1:
        # Guard: vLLM in DDP can be heavy; fallback to HF when world_size>1 unless explicitly desired
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        use_vllm = bool(getattr(args, 'hrpg_use_vllm', 0)) and (world_size == 1)
        if bool(getattr(args, 'hrpg_use_vllm', 0)) and world_size > 1 and gpu_id == 0:
            logger.warning('HRPG vLLM backend is not recommended with DDP>1; falling back to HF backend for training.')
        cfg = HRPGConfig(
            device=f'cuda:{gpu_id}',
            use_qwen_vl=True,
            qwen_model_name=(getattr(args, 'qwen_model', '') or 'Qwen/Qwen-VL-Chat'),
            use_vllm=use_vllm,
            vllm_gpu_mem=float(getattr(args, 'vllm_gpu_mem', 0.70)),
            vllm_max_tokens=int(getattr(args, 'vllm_max_tokens', 512)),
            vllm_temperature=float(getattr(args, 'vllm_temperature', 0.0)),
            vllm_top_k=int(getattr(args, 'vllm_top_k', -1)),
            max_boxes_from_vlm=int(getattr(args, 'hrpg_max_boxes', 5)),
            points_per_box=int(getattr(args, 'hrpg_points_per_box', 1)),
            prompt_image_size=(int(getattr(args, 'hrpg_prompt_h', 512)), int(getattr(args, 'hrpg_prompt_w', 512))),
        )
        try:
            hrpg_reasoner = QwenVLReasoner(model_name=cfg.qwen_model_name, device=cfg.device, cfg=cfg)
            if gpu_id == 0:
                logger.info(f'HRPG enabled. Backend={"vLLM" if cfg.use_vllm else "Transformers"}, model={cfg.qwen_model_name}')
        except Exception as e:
            hrpg_reasoner = None
            if gpu_id == 0:
                logger.warning(f'Failed to initialize HRPG reasoner: {e}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.)
    scheduler = MultiStepLR(optimizer, milestones=[35], gamma=0.1)
    best_acc = 0
    best_epoch = 0
    start_epoch = 0

    # Resume from checkpoint if provided
    if getattr(args, 'resume', None):
        resume_path = args.resume
        try:
            ckpt = torch.load(resume_path, map_location='cpu')
            if isinstance(ckpt, dict) and 'model' in ckpt:
                # Full checkpoint
                model.module.load_state_dict(ckpt['model'], strict=getattr(args, 'strict', False))
                if 'optimizer' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer'])
                if 'scheduler' in ckpt:
                    scheduler.load_state_dict(ckpt['scheduler'])
                if 'scaler' in ckpt:
                    try:
                        scaler.load_state_dict(ckpt['scaler'])
                    except Exception:
                        pass
                best_acc = ckpt.get('best_acc', best_acc)
                best_epoch = ckpt.get('best_epoch', best_epoch)
                start_epoch = ckpt.get('epoch', -1) + 1
                if gpu_id == 0:
                    logger.info(f"Resumed full checkpoint from {resume_path} at epoch {start_epoch}, best_acc={best_acc:.4f}, best_epoch={best_epoch}")
            else:
                # Model-only weights
                state_dict = ckpt
                if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                model.module.load_state_dict(state_dict, strict=getattr(args, 'strict', False))
                if gpu_id == 0:
                    logger.info(f"Loaded model weights from {resume_path} (model-only), starting from epoch 0")
        except Exception as e:
            if gpu_id == 0:
                logger.error(f"Failed to resume from {resume_path}: {e}")

    for epoch in range(start_epoch, max_epoch):
        train_sampler.set_epoch(epoch + 1)

        hrpg_kpp = int(getattr(args, 'hrpg_points_per_box', 1))
        train_loss, train_acc, train_f1, train_iou, train_loss1, train_loss2, tlogs = train(train_loader=train_loader,
                                                                                             criterion=criterion, model=model,
                                                                                             optimizer=optimizer,
                                                                                             gpu_id=gpu_id,
                                                                                             hrpg_reasoner=hrpg_reasoner,
                                                                                             hrpg_kpp=hrpg_kpp)
        if gpu_id == 0:
            logger.info(
                'Epoch:[{}/{}]\t train_loss={:.4f}\t train_ACC={:.8f}\t train_F1={:.8f}\t train_IoU={:.8f}\t loss1={:.4f}\t loss2={:.4f}\t temp_mean={:.4f}\t temp_std={:.4f}\t loss_ctr={:.4f}\t pos_count={:.2f}\t kept_ratio={:.2f}\t'.format(
                    epoch, max_epoch,
                    train_loss, train_acc,
                    train_f1, train_iou, train_loss1, train_loss2,
                    tlogs.get('temp_mean', 0.0), tlogs.get('temp_std', 0.0), tlogs.get('loss_ctr', 0.0), tlogs.get('pos_count', 0.0), tlogs.get('kept_ratio', 0.0)))

        val_acc, _, f1, iou = validate(val_loader=val_loader, model=model, gpu_id=gpu_id, hrpg_reasoner=hrpg_reasoner, hrpg_kpp=hrpg_kpp)
        if val_acc > best_acc and gpu_id == 0:
            best_acc = val_acc
            best_epoch = epoch
            ckp_name = 'epoch:{}_acc:{:.4f}_f1:{:.4f}.pth'.format(epoch, val_acc, f1)
            # Keep original model-only snapshot (for backward compatibility)
            torch.save(model.module.state_dict(), os.path.join(ckp_savepath, TITLE + ckp_name))
            # Save full best checkpoint for resume
            best_ckpt = {
                'epoch': epoch,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc': best_acc,
                'best_epoch': best_epoch,
            }
            torch.save(best_ckpt, os.path.join(ckp_savepath, f"{TITLE}_best.pth"))
        if gpu_id == 0:
            logger.info(
                'Epoch:[{}/{}]\t val_ACC={:.8f}\t  F1={:.8f}\t IoU={:.8f}\t best_epoch={}\t best_ACC={:.4f}\t'.format(
                    epoch,
                    max_epoch,
                    val_acc, f1, iou,
                    best_epoch,
                    best_acc))
        scheduler.step()

        # Always save last checkpoint (rank 0 only)
        if gpu_id == 0:
            last_ckpt = {
                'epoch': epoch,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc': best_acc,
                'best_epoch': best_epoch,
            }
            torch.save(last_ckpt, os.path.join(ckp_savepath, f"{TITLE}_last.pth"))


def train(train_loader, criterion, gpu_id, model, optimizer, hrpg_reasoner=None, hrpg_kpp: int = 1):
    model.train()

    epoch_loss = 0
    loss1_sum = 0
    loss2_sum = 0
    cm_all = np.zeros((23, 23))
    tp = 0
    fp = 0
    fn = 0

    # diagnostics accumulators (on device)
    sum_temp_mean = torch.tensor(0., device=gpu_id)
    sum_temp_std = torch.tensor(0., device=gpu_id)
    sum_loss_ctr = torch.tensor(0., device=gpu_id)
    sum_pos_count = torch.tensor(0., device=gpu_id)
    sum_kept = torch.tensor(0., device=gpu_id)
    sum_bs = torch.tensor(0., device=gpu_id)
    steps_t = torch.tensor(0., device=gpu_id)

    for imgs1, imgs2, q_str, type_str, answer_vec, mask_img, _, in tqdm(train_loader):
        imgs1 = imgs1.to(gpu_id).bfloat16()
        imgs2 = imgs2.to(gpu_id).bfloat16()
        answer_vec = answer_vec.to(gpu_id).bfloat16()
        mask_img = mask_img.to(gpu_id).bfloat16()
        # build supervised type indices
        device_ = torch.device('cuda', gpu_id) if torch.cuda.is_available() else torch.device('cpu')
        type_idx = torch.tensor([
            TYPE2IDX.get(t if t != 'change_ratio_types' else 'change_ratio', 0) for t in type_str
        ], device=device_, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)

        # Inject HRPG prompts if enabled
        if hrpg_reasoner is not None:
            try:
                in_h, in_w = model.module.MaskDecoder.prompt_decoder.input_image_size
            except Exception:
                in_h, in_w = 512, 512
            pts = build_hrpg_points(hrpg_reasoner, imgs1, imgs2, q_str, input_image_size=(in_h, in_w), k_points_per_box=hrpg_kpp)
            # move to device
            pts_dev = (pts[0].to(gpu_id), pts[1].to(gpu_id))
            model.module.use_external_prompts = True
            model.module.external_prompts = {'points': pts_dev}
        else:
            model.module.use_external_prompts = False
            model.module.external_prompts = None

        with torch.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
            out = model(imgs1, imgs2, q_str, mask_img, answer_vec, type_idx=type_idx)
            if isinstance(out, (list, tuple)) and len(out) == 7:
                pred, ans, target, loss, loss1, loss2, logd = out
            else:
                pred, ans, target, loss, loss1, loss2 = out
                logd = None

        pred = pred.flatten(1)
        target = target.flatten(1)
        pred = torch.sigmoid(pred)
        pred[pred < 0.35] = 0.
        pred[pred >= 0.35] = 1.

        pred = pred.cpu().float().numpy()
        target = target.cpu().float().numpy()

        tp += np.sum((pred == 1) & (target == 1))
        fp += np.sum((pred == 1) & (target == 0))
        fn += np.sum((pred == 0) & (target == 1))

        ans = ans.argmax(dim=1)
        answer_vec = answer_vec.argmax(dim=1)
        cm_all += confuse_matrix(ans, answer_vec, 23)

        epoch_loss += loss.item()
        loss1_sum += loss1.item()
        loss2_sum += loss2.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # accumulate diagnostics
        if logd is not None:
            sum_temp_mean += logd['temp_mean']
            sum_temp_std += logd['temp_std']
            sum_loss_ctr += logd['loss_ctr']
            sum_pos_count += logd['pos_count']
            sum_kept += logd['kept_samples']
            sum_bs += logd['batch_size']
            steps_t += 1.0

    cm_all = torch.from_numpy(cm_all).to(gpu_id)
    tp = torch.tensor(tp).to(gpu_id)
    fp = torch.tensor(fp).to(gpu_id)
    fn = torch.tensor(fn).to(gpu_id)

    dist.all_reduce(cm_all)
    dist.all_reduce(tp)
    dist.all_reduce(fp)
    dist.all_reduce(fn)
    # reduce diagnostics
    dist.all_reduce(sum_temp_mean)
    dist.all_reduce(sum_temp_std)
    dist.all_reduce(sum_loss_ctr)
    dist.all_reduce(sum_pos_count)
    dist.all_reduce(sum_kept)
    dist.all_reduce(sum_bs)
    dist.all_reduce(steps_t)

    cm_all = cm_all.cpu().numpy()
    tp = tp.cpu().numpy()
    fp = fp.cpu().numpy()
    fn = fn.cpu().numpy()

    # build logs averaged across ranks
    steps_g = steps_t.item() if steps_t.item() > 0 else 1.0
    logs = {
        'temp_mean': (sum_temp_mean / steps_g).item() if steps_g > 0 else 0.0,
        'temp_std': (sum_temp_std / steps_g).item() if steps_g > 0 else 0.0,
        'loss_ctr': (sum_loss_ctr / steps_g).item() if steps_g > 0 else 0.0,
        'pos_count': (sum_pos_count / steps_g).item() if steps_g > 0 else 0.0,
        'kept_ratio': (sum_kept / (sum_bs + 1e-6)).item() if sum_bs.item() > 0 else 0.0,
    }

    return (epoch_loss, cm_all.diagonal().sum() / cm_all.sum(), 2 * tp / (2 * tp + fp + fn),
            tp / (tp + fp + fn), loss1_sum, loss2_sum, logs)


def validate(val_loader, gpu_id, model, hrpg_reasoner=None, hrpg_kpp: int = 1):
    cm_all = np.zeros((23, 23))
    model.eval()
    tp = 0
    fp = 0
    fn = 0

    for imgs1, imgs2, q_str, type_str, answer_vec, mask_img, _,in val_loader:
        imgs1 = imgs1.to(gpu_id)
        imgs2 = imgs2.to(gpu_id)
        answer_vec = answer_vec.to(gpu_id)
        mask_img = mask_img.to(gpu_id)

        # Inject HRPG prompts if enabled
        if hrpg_reasoner is not None:
            try:
                in_h, in_w = model.module.MaskDecoder.prompt_decoder.input_image_size
            except Exception:
                in_h, in_w = 512, 512
            pts = build_hrpg_points(hrpg_reasoner, imgs1, imgs2, q_str, input_image_size=(in_h, in_w), k_points_per_box=hrpg_kpp)
            pts_dev = (pts[0].to(gpu_id), pts[1].to(gpu_id))
            model.module.use_external_prompts = True
            model.module.external_prompts = {'points': pts_dev}
        else:
            model.module.use_external_prompts = False
            model.module.external_prompts = None

        with torch.no_grad():
            pred, ans = model(imgs1, imgs2, q_str, mask_img)

        pred = pred.flatten(1)
        target = mask_img.flatten(1)
        pred[pred < 0.35] = 0.
        pred[pred >= 0.35] = 1.

        pred = pred.cpu().float().numpy()
        target = target.cpu().float().numpy()

        tp += np.sum((pred == 1) & (target == 1))
        fp += np.sum((pred == 1) & (target == 0))
        fn += np.sum((pred == 0) & (target == 1))

        ans = ans.argmax(dim=1)
        answer_vec = answer_vec.argmax(dim=1)
        cm_all += confuse_matrix(ans, answer_vec, 23)

    cm_all = torch.from_numpy(cm_all).to(gpu_id)
    tp = torch.tensor(tp).to(gpu_id)
    fp = torch.tensor(fp).to(gpu_id)
    fn = torch.tensor(fn).to(gpu_id)

    dist.all_reduce(cm_all)
    dist.all_reduce(tp)
    dist.all_reduce(fp)
    dist.all_reduce(fn)

    cm_all = cm_all.cpu().numpy()
    tp = tp.cpu().numpy()
    fp = fp.cpu().numpy()
    fn = fn.cpu().numpy()

    return cm_all.diagonal().sum() / cm_all.sum(), cm_all, 2 * tp / (2 * tp + fp + fn), tp / (tp + fp + fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from (supports full or model-only)')
    parser.add_argument('--strict', action='store_true', help='Strictly load state dicts when resuming')
    parser.add_argument('--ckp_dir', type=str, default='', help='Directory to save checkpoints (default: ./ckps/VisTA)')
    parser.add_argument('--log_dir', type=str, default='', help='Directory to save logs (default: ./logs)')
    # Hyperparameters for new features
    parser.add_argument('--sample_ratio', type=float, default=0.7, help='sample-level hard mining keep ratio')
    parser.add_argument('--pixel_hard_ratio', type=float, default=0.25, help='pixel-level OHEM ratio per sample')
    parser.add_argument('--contrast_weight', type=float, default=0.05, help='weight for contrastive loss')
    parser.add_argument('--temp_reg_weight', type=float, default=0.01, help='weight for temperature regularization')
    parser.add_argument('--type_weight', type=float, default=0.1, help='weight for type supervision')
    parser.add_argument('--consistency_weight', type=float, default=0.05, help='weight for no-change consistency')
    parser.add_argument('--contrast_tau', type=float, default=0.07, help='temperature for InfoNCE')
    parser.add_argument('--enable_contrast', type=int, default=1, help='enable contrastive alignment (1/0)')
    parser.add_argument('--enable_uncertainty', type=int, default=1, help='enable uncertainty temperature (1/0)')
    parser.add_argument('--enable_hard_mining', type=int, default=1, help='enable sample-level hard mining (1/0)')
    # HRPG / Qwen-VL integration (default off)
    parser.add_argument('--use_hrpg', type=int, default=0, help='enable HRPG prompts (1/0)')
    parser.add_argument('--qwen_model', type=str, default='', help='Qwen-VL HF name or local path (e.g., Qwen/Qwen2-VL-7B-Instruct)')
    parser.add_argument('--hrpg_use_vllm', type=int, default=0, help='use vLLM backend for Qwen (1/0); recommend 0 for DDP training')
    parser.add_argument('--vllm_gpu_mem', type=float, default=0.70, help='vLLM gpu_memory_utilization')
    parser.add_argument('--vllm_max_tokens', type=int, default=512)
    parser.add_argument('--vllm_temperature', type=float, default=0.0)
    parser.add_argument('--vllm_top_k', type=int, default=-1)
    parser.add_argument('--hrpg_points_per_box', type=int, default=1, help='points per box when converting VLM boxes to points')
    parser.add_argument('--hrpg_max_boxes', type=int, default=5, help='max boxes requested from VLM')
    parser.add_argument('--hrpg_prompt_h', type=int, default=512)
    parser.add_argument('--hrpg_prompt_w', type=int, default=512)
    args = parser.parse_args()
    main(args)
