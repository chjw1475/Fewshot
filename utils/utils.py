from tqdm import tqdm
import numpy as np
import torch

def preprocess_data(args, loader):
    """
    preprocess data from pickle
    :param args: args
    :param loader: loader
    :return: [support, s_oneshot, query, q_oneshot]
    """
    # sample data for next batch
    support, s_labels, query, q_labels, unlabel = loader.next_data(args.n_way, args.n_shot, args.n_query)
    support = np.reshape(support, (support.shape[0] * support.shape[1],) + support.shape[2:])
    support = torch.from_numpy(np.transpose(support, (0, 3, 1, 2)))
    query = np.reshape(query, (query.shape[0] * query.shape[1],) + query.shape[2:])
    query = torch.from_numpy(np.transpose(query, (0, 3, 1, 2)))
    s_labels = torch.from_numpy(np.reshape(s_labels, (-1,)))
    q_labels = torch.from_numpy(np.reshape(q_labels, (-1,)))
    s_labels = s_labels.type(torch.LongTensor)
    q_labels = q_labels.type(torch.LongTensor)
    s_onehot = torch.zeros(args.n_way * args.n_shot, args.n_way).scatter_(1, s_labels.view(-1, 1), 1)
    q_onehot = torch.zeros(args.n_way * args.n_query, args.n_way).scatter_(1, q_labels.view(-1, 1), 1)

    inputs = [support.cuda(0), s_onehot.cuda(0), query.cuda(0), q_onehot.cuda(0)]

    return inputs

def train(args, ep, model, model_scheduler, loader_train):
    """
    train N episods
    :param args: args
    :param ep: current epoch number
    :param model: model
    :param model_scheduler: model optimizer scheduler
    :param loader_train: loader of train set
    :return: train loss, train acc(after update), train reverse loss(if available)
    """
    tr_loss = []
    tr_acc = []
    tr_loss_reverse = []

    for epi in tqdm(range(args.n_episodes), desc='train epoch:{}'.format(ep+1)):
        model_scheduler.step(ep * args.n_episodes + epi)
        # set train mode
        model.train()

        inputs = preprocess_data(args, loader_train)

        if args.alg in ['cycle_1', 'cycle_2']:
            emb_s, emb_q, emb_ss, emb_qq = model(inputs)
            loss, reverse_loss = model.cal_loss(emb_s, emb_q, emb_ss, emb_qq)
            model.update(loss + reverse_loss)
            tr_loss_reverse.append(reverse_loss.item())
        elif args.alg == 'cycle_3':
            emb_s, emb_q, emb_ss, emb_qq = model(inputs)
            loss, _ = model.cal_loss(emb_s, emb_q, emb_ss, emb_qq)
            model.update(loss)
            emb_s, emb_q, emb_ss, emb_qq = model(inputs)
            _, reverse_loss = model.cal_loss(emb_s, emb_q, emb_ss, emb_qq)
            model.update(reverse_loss)
            tr_loss_reverse.append(reverse_loss.item())
        else:
            emb_s, emb_q = model(inputs)
            loss = model.cal_loss(emb_s, emb_q)
            model.update(loss)

        acc = model.get_acc(inputs)
        tr_loss.append(loss.item())
        tr_acc.append(acc.item())

    if len(tr_loss_reverse) > 0:
        return tr_loss, tr_acc, tr_loss_reverse
    else:
        return tr_loss, tr_acc

def evaluation(args, ep, model, loader, dataset='val'):
    """
    evaludate validation N episods
    :param args: args
    :param ep: current epoch number
    :param model: model
    :param loader: loader of validation set
    :param dataset: 'val' or 'test'
    :return: val loss, val acc(after update), val reverse loss(if available)
    """
    eval_loss = []
    eval_acc = []
    eval_loss_reverse = []
    n_episodes = {'val': args.n_episodes, 'test':args.n_test_episodes}[dataset]

    for _ in tqdm(range(n_episodes), desc='{} epoch:{}'.format(dataset, ep+1)):
        # set eval mode
        model.eval()

        with torch.no_grad():
            # sample data for next batch
            inputs = preprocess_data(args, loader)
            if args.alg in ['cycle_1', 'cycle_2', 'cycle_3']:
                emb_s, emb_q, emb_ss, emb_qq = model(inputs)
                loss, reverse_loss = model.cal_loss(emb_s, emb_q, emb_ss, emb_qq)
                eval_loss_reverse.append(reverse_loss.item())
            else:
                emb_s, emb_q = model(inputs)
                loss = model.cal_loss(emb_s, emb_q)
            acc = model.get_acc([emb_s, emb_q])

        eval_loss.append(loss.item())
        eval_acc.append(acc.item())
    if len(eval_loss_reverse) > 0:
        return eval_loss, eval_acc, eval_loss_reverse
    else:
        return eval_loss, eval_acc