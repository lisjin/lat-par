import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ogb.graphproppred import Evaluator
from torch.utils.data import DataLoader

from data import Vocab, MyDataset, STR, END, CLS, SEL, rCLS, sparsify_batch, worker_init_fn
from generator import Generator
from extract import LexicalMap
from adam import AdamWeightDecayOptimizer
from utils import move_to_cuda
import argparse, os
import datetime
import random
import re

RAND_N = 19940117

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_file', type=str, default=None)
    parser.add_argument('--log_file', type=str, default='train.log')

    # concept encoders
    parser.add_argument('--concept_dim', type=int)

    # relation encoder
    parser.add_argument('--rel_dim', type=int)
    parser.add_argument('--nt_size', type=int)

    # core architecture
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--ff_embed_dim', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--graph_layers', type=int)

    # dropout/unk
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--weight_decay', type=float)

    # IO
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--train_forests', type=str)
    parser.add_argument('--train_sep2frags', type=str)
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--dev_batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--eval_every', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--detect_anomaly', action='store_true')

    # distributed training
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--MASTER_ADDR', type=str)
    parser.add_argument('--MASTER_PORT', type=str)
    parser.add_argument('--start_rank', type=int)

    return parser.parse_args()

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size

def update_lr(optimizer, embed_size, steps, warmup_steps):
    for param_group in optimizer.param_groups:
        param_group['lr'] = embed_size**-0.5 * min(steps**-0.5, steps*(warmup_steps**-1.5))

def feed_batch(model, batch):
    sparsify_batch(batch)
    batch = move_to_cuda(batch, model.device)
    loss, pred, target = model(batch)
    loss_value = loss.item()
    return loss, loss_value, pred, target

def print_eval(args, batches_acm, loss_acm, model, dev_loader, log_f, optimizer,
        epoch, best_roc, evaluator):
    if batches_acm % args.print_every == -1 % args.print_every:
        avg_loss = loss_acm / batches_acm
        print (f'epoch {epoch}, batch {batches_acm}, loss {avg_loss:.3f}')
        model.train()
    if batches_acm > args.warmup_steps and batches_acm % args.eval_every == -1\
            % args.eval_every:
        model.eval()
        dev_loss = 0.
        dev_n = 0
        pred_lst = []
        target_lst = []
        for batch in dev_loader:
            _, loss, pred, target = feed_batch(model, batch)
            pred_lst.append(pred.detach())
            target_lst.append(target.detach().unsqueeze(-1))
            dev_loss += loss
            dev_n += 1
        dev_loss /= dev_n
        dev_roc = evaluator.eval({'y_pred': torch.cat(pred_lst),
            'y_true': torch.cat(target_lst)})['rocauc'].item()
        log_f.write(f'{epoch} {dev_loss:.3f} {dev_roc:.4f}\n')
        log_f.flush()
        if dev_roc > best_roc:
            best_roc = dev_roc
            ckpt_files = get_ckpt_files(args.ckpt)
            if len(ckpt_files) > 1:
                oldest = min(ckpt_files, key=os.path.getctime)
                os.remove(oldest)
            torch.save({
                'args': args,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'batches_acm': batches_acm,
                'loss_acm': loss_acm,
                'roc': best_roc},
                os.path.join(args.ckpt, f'batch{batches_acm:06d}_epoch{epoch:03d}'))
        model.train()
    return best_roc

def main(args, local_rank):
    torch.manual_seed(RAND_N)
    torch.cuda.manual_seed_all(RAND_N)
    random.seed(RAND_N)

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    model = Generator(args.concept_dim, args.rel_dim,
            args.nt_size, args.embed_dim,
            args.ff_embed_dim, args.num_heads, args.dropout,
            args.graph_layers, args.pretrained_file, device).cuda(device)

    if args.world_size > 1:
        rnd = RAND_N + dist.get_rank()
        torch.manual_seed(rnd)
        torch.cuda.manual_seed_all(rnd)
        random.seed(rnd)

    train_data = MyDataset(args.train_data, args.train_forests, args.train_sep2frags, args.train_batch_size, split='train')
    dev_data = MyDataset(args.train_data, args.train_forests, args.train_sep2frags, args.dev_batch_size, split='valid')

    weight_decay_params = []
    no_weight_decay_params = []
    for name, param in model.named_parameters():
        if name.endswith('bias') or 'layer_norm' in name:
            no_weight_decay_params.append(param)
        else:
            weight_decay_params.append(param)
    grouped_params = [{'params':weight_decay_params, 'weight_decay': args.weight_decay},
                        {'params':no_weight_decay_params, 'weight_decay':0.}]
    optimizer = AdamWeightDecayOptimizer(grouped_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-6)
    evaluator = Evaluator(name=args.train_data)

    epoch = 0
    batches_acm, loss_acm = 0, 0
    discarded_batches_acm = 0
    best_roc = 0.
    if hasattr(args, 'last_ckpt'):
        map_location = {'cuda:0': 'cuda:{}'.format(local_rank)}
        ckpt = torch.load(args.last_ckpt, map_location=map_location)
        model.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt['epoch']
        batches_acm = ckpt['batches_acm']
        loss_acm = ckpt['loss_acm']
        best_roc = ckpt['roc']
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        del ckpt

    train_loader = DataLoader(train_data, worker_init_fn=worker_init_fn,
            num_workers=args.num_workers, batch_size=None)
    dev_loader = DataLoader(dev_data, worker_init_fn=worker_init_fn,
            num_workers=1, batch_size=None)

    with open(os.path.join(args.ckpt, args.log_file), 'a') as log_f:
        if local_rank == 0:
            log_f.write(f'Start time: {datetime.datetime.now()}\n')
            log_f.flush()
        for epoch in range(epoch, args.epochs):
            model.train()
            for batch in train_loader:
                loss, loss_value, _, _ = feed_batch(model, batch)
                if batches_acm > args.warmup_steps and loss_value > 5.*\
                        (loss_acm / batches_acm):
                    discarded_batches_acm += 1
                    print ('abnormal', loss_value)
                    continue
                loss_acm += loss_value
                batches_acm += 1
                loss.backward()
                if args.world_size > 1:
                    average_gradients(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                update_lr(optimizer, args.embed_dim, batches_acm, args.warmup_steps)
                optimizer.step()
                optimizer.zero_grad()
                if args.world_size == 1 or (dist.get_rank() == 0):
                    del batch
                    best_roc = print_eval(args, batches_acm, loss_acm, model,
                            dev_loader, log_f, optimizer, epoch, best_roc,
                            evaluator)
        log_f.write(f'Finish time: {datetime.datetime.now()}\n')

def init_processes(args, local_rank, backend='nccl'):
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    dist.init_process_group(backend, rank=args.start_rank + local_rank,
            world_size=args.world_size)
    main(args, local_rank)

def get_ckpt_files(ckpt_path):
    return [os.path.join(ckpt_path, x) for x in os.listdir(ckpt_path) if\
            re.search(r'\d+$', x)]

if __name__ == "__main__":
    args = parse_config()
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if os.path.exists(args.ckpt):
        files = get_ckpt_files(args.ckpt)
        if files:
            args.last_ckpt = max(files, key=os.path.getctime)
    else:
        os.mkdir(args.ckpt)

    if args.world_size == 1:
        main(args, 0)
        exit(0)
    args.train_batch_size = args.train_batch_size / args.world_size
    processes = []
    for rank in range(args.gpus):
        p = mp.Process(target=init_processes, args=(args, rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

