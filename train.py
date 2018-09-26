import argparse
import os.path
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from torch.autograd import Variable
from tqdm import tqdm

import models
import utils
from datasets import vqa_dataset


def train(model, loader, optimizer, tracker, epoch, split):
    model.train()

    tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    tq = tqdm(loader, desc='{} E{:03d}'.format(split, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(split), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(split), tracker_class(**tracker_params))
    log_softmax = nn.LogSoftmax(dim=1).cuda()

    for item in tq:
        v = item['visual']
        q = item['question']
        a = item['answer']
        q_length = item['q_length']

        v = Variable(v.cuda(async=True))
        q = Variable(q.cuda(async=True))
        a = Variable(a.cuda(async=True))
        q_length = Variable(q_length.cuda(async=True))

        out = model(v, q, q_length)

        # This is the Soft-loss described in https://arxiv.org/pdf/1708.00584.pdf

        nll = -log_softmax(out)

        loss = (nll * a / 10).sum(dim=1).mean()
        acc = utils.vqa_accuracy(out.data, a.data).cpu()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_tracker.append(loss.item())
        acc_tracker.append(acc.mean())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))


def evaluate(model, loader, tracker, epoch, split):
    model.eval()
    tracker_class, tracker_params = tracker.MeanMonitor, {}

    predictions = []
    samples_ids = []
    accuracies = []

    tq = tqdm(loader, desc='{} E{:03d}'.format(split, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(split), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(split), tracker_class(**tracker_params))
    log_softmax = nn.LogSoftmax(dim=1).cuda()

    with torch.no_grad():
        for item in tq:
            v = item['visual']
            q = item['question']
            a = item['answer']
            sample_id = item['sample_id']
            q_length = item['q_length']

            v = Variable(v.cuda(async=True))
            q = Variable(q.cuda(async=True))
            a = Variable(a.cuda(async=True))
            q_length = Variable(q_length.cuda(async=True))

            out = model(v, q, q_length)

            # This is the Soft-loss described in https://arxiv.org/pdf/1708.00584.pdf

            nll = -log_softmax(out)

            loss = (nll * a / 10).sum(dim=1).mean()
            acc = utils.vqa_accuracy(out.data, a.data).cpu()

            # save predictions of this batch
            _, answer = out.data.cpu().max(dim=1)

            predictions.append(answer.view(-1))
            accuracies.append(acc.view(-1))
            # Sample id is necessary to obtain the mapping sample-prediction
            samples_ids.append(sample_id.view(-1).clone())

            loss_tracker.append(loss.item())
            acc_tracker.append(acc.mean())
            fmt = '{:.4f}'.format
            tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

        predictions = list(torch.cat(predictions, dim=0))
        accuracies = list(torch.cat(accuracies, dim=0))
        samples_ids = list(torch.cat(samples_ids, dim=0))

    eval_results = {
        'answers': predictions,
        'accuracies': accuracies,
        'samples_ids': samples_ids,
        'avg_accuracy': acc_tracker.mean.value,
        'avg_loss': loss_tracker.mean.value
    }

    return eval_results


def main():
    # Load config yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config', default='config/default.yaml', type=str,
                        help='path to a yaml config file')
    args = parser.parse_args()

    if args.path_config is not None:
        with open(args.path_config, 'r') as handle:
            config = yaml.load(handle)

    # generate log directory
    dir_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    path_log_dir = os.path.join(config['logs']['dir_logs'], dir_name)

    if not os.path.exists(path_log_dir):
        os.makedirs(path_log_dir)

    print('Model logs will be saved in {}'.format(path_log_dir))

    cudnn.benchmark = True

    # Generate datasets and loaders
    train_loader = vqa_dataset.get_loader(config, split='train')
    val_loader = vqa_dataset.get_loader(config, split='val')

    model = nn.DataParallel(models.Model(config, train_loader.dataset.num_tokens)).cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 config['training']['lr'])

    # Load model weights if necessary
    if config['model']['pretrained_model'] is not None:
        print("Loading Model from %s" % config['model']['pretrained_model'])
        log = torch.load(config['model']['pretrained_model'])
        dict_weights = log['weights']
        model.load_state_dict(dict_weights)

    tracker = utils.Tracker()

    min_loss = 10
    max_accuracy = 0

    path_best_accuracy = os.path.join(path_log_dir, 'best_accuracy_log.pth')
    path_best_loss = os.path.join(path_log_dir, 'best_loss_log.pth')

    for i in range(config['training']['epochs']):

        train(model, train_loader, optimizer, tracker, epoch=i, split=config['training']['train_split'])
        # If we are training on the train split (and not on train+val) we can evaluate on val
        if config['training']['train_split'] == 'train':
            eval_results = evaluate(model, val_loader, tracker, epoch=i, split='val')

            # save all the information in the log file
            log_data = {
                'epoch': i,
                'tracker': tracker.to_dict(),
                'config': config,
                'weights': model.state_dict(),
                'eval_results': eval_results,
                'vocabs': train_loader.dataset.vocabs,
            }

            # save logs for min validation loss and max validation accuracy
            if eval_results['avg_loss'] < min_loss:
                torch.save(log_data, path_best_loss)  # save model
                min_loss = eval_results['avg_loss']  # update min loss value

            if eval_results['avg_accuracy'] > max_accuracy:
                torch.save(log_data, path_best_accuracy)  # save model
                max_accuracy = eval_results['avg_accuracy']  # update max accuracy value

    # Save final model
    log_data = {
        'tracker': tracker.to_dict(),
        'config': config,
        'weights': model.state_dict(),
        'vocabs': train_loader.dataset.vocabs,
    }

    path_final_log = os.path.join(path_log_dir, 'final_log.pth')
    torch.save(log_data, path_final_log)


if __name__ == '__main__':
    main()
