import os
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.DataNew import DatasetNew
from model.VisTA import VisTA

try:
    from openpyxl import Workbook
except Exception:
    Workbook = None

# Label dictionaries (must align with training)
w2v_dict = {
    '0': 0, '0_to_10': 1, '10_to_20': 2, '20_to_30': 3, '30_to_40': 4, '40_to_50': 5, '50_to_60': 6,
    '60_to_70': 7, '70_to_80': 8, '80_to_90': 9, '90_to_100': 10, 'NVG_surface': 11, 'buildings': 12,
    'low_vegetation': 13, 'playgrounds': 14, 'trees': 15, 'water': 16, 'no': 17, 'yes': 18, 'green_house': 19,
    'road': 20, 'bridge': 21, 'others': 22
}

v2w_dict = {v: k for k, v in w2v_dict.items()}

# Paper display order: CN, CtW, CfW, IN, DN, LC, SC, CR
paper_order = [
    'change_or_not', 'change_to_what', 'change_from_what',
    'increase_or_not', 'decrease_or_not', 'largest_change', 'smallest_change', 'change_ratio'
]


def confuse_matrix(pre, label, n_class):
    pre = pre.cpu().numpy()
    label = label.cpu().numpy()
    cm = np.bincount(label * n_class + pre, minlength=n_class ** 2).reshape(n_class, n_class)
    return cm


def run_eval(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    model = VisTA()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    # Load checkpoint
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    state = torch.load(args.ckpt, map_location=device)
    incompatible = model.module.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print('[load_state_dict] missing:', len(incompatible.missing_keys), 'unexpected:', len(incompatible.unexpected_keys))

    # Dataset / Loader
    json_p = os.path.join(args.test_root, 'json', 'Test.json')
    img_dir = os.path.join(args.test_root, 'image')
    mask_dir = os.path.join(args.test_root, 'mask')
    val_dataset = DatasetNew(json_p, img_dir, mask_dir)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model.eval()

    cm_all = np.zeros((23, 23))
    cm_nochange = np.zeros((2, 2))

    # textual metrics accumulators
    type_sum = {
        'change_ratio': [0, 0], 'change_or_not': [0, 0], 'change_to_what': [0, 0],
        'increase_or_not': [0, 0], 'decrease_or_not': [0, 0], 'smallest_change': [0, 0], 'largest_change': [0, 0],
        'change_from_what': [0, 0],
    }

    ans_type_sum = {
        'change_ratio': {k: [0, 0] for k in ['0','0_to_10','10_to_20','20_to_30','30_to_40','40_to_50','50_to_60','60_to_70','70_to_80','80_to_90','90_to_100']},
        'change_or_not': {k: [0, 0] for k in ['yes','no']},
        'change_to_what': {k: [0, 0] for k in ['NVG_surface','buildings','low_vegetation','playgrounds','trees','water','green_house','road','bridge','others']},
        'increase_or_not': {k: [0, 0] for k in ['yes','no']},
        'decrease_or_not': {k: [0, 0] for k in ['yes','no']},
        'smallest_change': {k: [0, 0] for k in ['NVG_surface','buildings','low_vegetation','playgrounds','trees','water','green_house','road','bridge','others']},
        'largest_change': {k: [0, 0] for k in ['NVG_surface','buildings','low_vegetation','playgrounds','trees','water','green_house','road','bridge','others']},
        'change_from_what': {k: [0, 0] for k in ['NVG_surface','buildings','low_vegetation','playgrounds','trees','water','green_house','road','bridge','others']},
    }

    # visual metrics accumulators (oIoU per type)
    type_cm = {  # 0: TP, 1: FP, 2: FN
        'change_ratio': [0, 0, 0], 'change_or_not': [0, 0, 0], 'change_to_what': [0, 0, 0],
        'increase_or_not': [0, 0, 0], 'decrease_or_not': [0, 0, 0], 'smallest_change': [0, 0, 0],
        'largest_change': [0, 0, 0], 'change_from_what': [0, 0, 0],
    }

    # mIoU accumulators
    type_miou_sum = {k: 0.0 for k in type_cm.keys()}
    type_miou_count = {k: 0 for k in type_cm.keys()}
    overall_miou_sum = 0.0
    overall_miou_count = 0

    tp = 0
    fp = 0
    fn = 0

    for imgs1, imgs2, q_str, type_str, answer_vec, mask_img, _ in tqdm(val_loader):
        imgs1 = imgs1.to(device)
        imgs2 = imgs2.to(device)
        answer_vec = answer_vec.to(device)
        mask_img = mask_img.to(device)

        with torch.no_grad():
            pred, ans = model(imgs1, imgs2, q_str, mask_img)

        # threshold mask
        pred[pred < args.threshold] = 0
        pred[pred >= args.threshold] = 1
        img_out = pred

        # no-change confusion (binary)
        pred_img_max = np.max(img_out.cpu().numpy(), axis=(1, 2)).flatten()
        mask_img_max = np.max(mask_img.cpu().numpy(), axis=(2, 3)).flatten()
        cm_nochange[0][0] += np.sum((pred_img_max == 1) & (mask_img_max == 1))
        cm_nochange[0][1] += np.sum((pred_img_max == 1) & (mask_img_max == 0))
        cm_nochange[1][0] += np.sum((pred_img_max == 0) & (mask_img_max == 1))
        cm_nochange[1][1] += np.sum((pred_img_max == 0) & (mask_img_max == 0))

        # flatten for IoU metrics
        pred_f = pred.flatten(1).cpu().numpy()
        target_f = mask_img.flatten(1).cpu().numpy()

        tp += np.sum((pred_f == 1) & (target_f == 1))
        fp += np.sum((pred_f == 1) & (target_f == 0))
        fn += np.sum((pred_f == 0) & (target_f == 1))

        # per-sample IoU for mIoU
        tp_per = ((pred_f == 1) & (target_f == 1)).sum(axis=1)
        fp_per = ((pred_f == 1) & (target_f == 0)).sum(axis=1)
        fn_per = ((pred_f == 0) & (target_f == 1)).sum(axis=1)
        denom = (tp_per + fp_per + fn_per).astype(np.float64)
        denom[denom == 0] = 1.0
        iou_per = tp_per / denom
        overall_miou_sum += float(iou_per.sum())
        overall_miou_count += iou_per.shape[0]

        # textual confusion
        ans_idx = ans.argmax(dim=1)
        gt_idx = answer_vec.argmax(dim=1)
        cm_all += confuse_matrix(ans_idx, gt_idx, 23)

        # accumulators per type
        for i in range(len(type_str)):
            t = type_str[i]
            if t == 'change_ratio_types':
                t = 'change_ratio'
            # visual oIoU per type
            type_cm[t][0] += np.sum((pred_f[i] == 1) & (target_f[i] == 1))
            type_cm[t][1] += np.sum((pred_f[i] == 1) & (target_f[i] == 0))
            type_cm[t][2] += np.sum((pred_f[i] == 0) & (target_f[i] == 1))
            # per-type mIoU
            type_miou_sum[t] += float(iou_per[i])
            type_miou_count[t] += 1

        # textual per-type accuracy
        for i in range(len(type_str)):
            t = type_str[i]
            if t == 'change_ratio_types':
                t = 'change_ratio'
            type_sum[t][1] += 1
            gt_word = v2w_dict[int(gt_idx[i].cpu())]
            ans_type_sum[t][gt_word][1] += 1
            if ans_idx[i] == gt_idx[i]:
                type_sum[t][0] += 1
                ans_type_sum[t][gt_word][0] += 1

    # textual metrics
    OA = cm_all.diagonal().sum() / cm_all.sum()
    textual_acc_types = [(type_sum[t][0] / (type_sum[t][1] + 1e-6)) for t in paper_order]
    AA = sum(textual_acc_types) / len(textual_acc_types)

    # visual metrics
    overall_oIoU = tp / (tp + fp + fn + 1e-6)
    overall_mIoU = overall_miou_sum / max(1, overall_miou_count)

    visual_oIoU_types = []
    visual_mIoU_types = []
    for t in paper_order:
        tp_t, fp_t, fn_t = type_cm[t]
        denom_t = (tp_t + fp_t + fn_t)
        o_iou_t = (tp_t / denom_t) if denom_t > 0 else 0.0
        visual_oIoU_types.append(o_iou_t)
        m_iou_t = (type_miou_sum[t] / type_miou_count[t]) if type_miou_count[t] > 0 else 0.0
        visual_mIoU_types.append(m_iou_t)

    # Export to Excel (paper-like)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    xlsx_path = args.xlsx if args.xlsx else f'results_{timestamp}.xlsx'
    csv_path = args.csv if args.csv else f'results_{timestamp}.csv'

    headers_text = ['Method', 'Backbone', 'CN', 'CtW', 'CfW', 'IN', 'DN', 'LC', 'SC', 'CR', 'AA', 'OA']
    row_text = [args.method_label, args.backbone_label] + [round(v * 100, 2) for v in textual_acc_types] + [round(AA * 100, 2), round(OA * 100, 2)]

    headers_vis = ['Method', 'Backbone', 'CN', 'CtW', 'CfW', 'IN', 'DN', 'LC', 'SC', 'CR', 'mIoU', 'oIoU']
    row_vis = [
        args.method_label, args.backbone_label,
        *[round(v * 100, 2) for v in visual_oIoU_types],
        round(overall_mIoU * 100, 2), round(overall_oIoU * 100, 2)
    ]

    try:
        if Workbook is None:
            raise ImportError('openpyxl not available')
        wb = Workbook()
        ws = wb.active
        ws.title = 'Results'
        ws.append(headers_text)
        ws.append(row_text)
        ws.append([])
        ws.append(headers_vis)
        ws.append(row_vis)

        ws2 = wb.create_sheet('Details')
        ws2.append(['Type', 'mIoU', 'oIoU'])
        for i, t in enumerate(paper_order):
            ws2.append([t, round(visual_mIoU_types[i] * 100, 2), round(visual_oIoU_types[i] * 100, 2)])
        ws2.append(['Overall', round(overall_mIoU * 100, 2), round(overall_oIoU * 100, 2)])

        wb.save(xlsx_path)
        print(f'Excel results saved to: {xlsx_path}')
    except Exception as e:
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers_text)
            writer.writerow(row_text)
            writer.writerow([])
            writer.writerow(headers_vis)
            writer.writerow(row_vis)
            writer.writerow([])
            writer.writerow(['Type', 'mIoU', 'oIoU'])
            for i, t in enumerate(paper_order):
                writer.writerow([t, round(visual_mIoU_types[i] * 100, 2), round(visual_oIoU_types[i] * 100, 2)])
            writer.writerow(['Overall', round(overall_mIoU * 100, 2), round(overall_oIoU * 100, 2)])
        print(f'openpyxl not available ({e}), CSV saved to: {csv_path}')

    # Optionally save confusion matrix like original test.py
    if args.save_cm:
        with open(args.save_cm, 'w') as f:
            for i in range(23):
                for j in range(23):
                    f.write(str(int(cm_all[i][j])) + ',')
                f.write('\n')
        print(f'Confusion matrix saved to: {args.save_cm}')

    # Print summary for console
    print('Textual -> AA: {:.4f} OA: {:.4f}'.format(AA, OA))
    print('Visual  -> mIoU: {:.4f} oIoU: {:.4f}'.format(overall_mIoU, overall_oIoU))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default='./ckps/VisTA/VisTa.pth', help='checkpoint path')
    p.add_argument('--test_root', type=str, default='./test/', help='test root containing json/image/mask')
    p.add_argument('--batch_size', type=int, default=48)
    p.add_argument('--num_workers', type=int, default=16)
    p.add_argument('--threshold', type=float, default=0.35)
    p.add_argument('--method_label', type=str, default='VisTA Ours')
    p.add_argument('--backbone_label', type=str, default='Res-101')
    p.add_argument('--xlsx', type=str, default='')
    p.add_argument('--csv', type=str, default='')
    p.add_argument('--save_cm', type=str, default='cm.csv')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_eval(args)
