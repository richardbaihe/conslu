import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn_crfsuite import metrics
from sklearn.metrics import f1_score,accuracy_score,recall_score

from utils import logger
from model import SDEN, Seq2Seq, SDEN_plus, MemNet,MemNet_plus
from utils.data_utils import data_loader,pad_to_batch,pad_to_batch_slm

model_dic = {'sden': SDEN,
             's2s': Seq2Seq,
             'sden_plus': SDEN_plus,
             'memnet': MemNet,
             'memnet_plus': MemNet_plus}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(24022019010102019)
torch.cuda.manual_seed_all(24022019010102019)

def train_multitask(model, train_data, dev_data, config,test_data=None):
    log = logger.Logger(config.save_path)
    config.best_score=-100
    train_data_1, train_data_2 = train_data
    dev_data_1, dev_data_2 = dev_data
    test_data_1, test_data_2 = test_data
    # slm_num = 0
    # slm_pos = torch.tensor(0.0)
    # for data in train_data_2:
    #     slm_num += data[-1].shape[-1]
    #     slm_pos += torch.sum(data[-1]).type_as(torch.tensor(0.3))
    # neg_weight = slm_pos / slm_num
    # pos_weight = 1 - neg_weight
    # slm_loss_weight = torch.tensor([neg_weight, pos_weight]).cuda()

    slm_loss = nn.CrossEntropyLoss()#slm_loss_weight)
    slot_loss_function = nn.CrossEntropyLoss(ignore_index=0)
    intent_loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1, milestones=[config.epochs // 4, config.epochs // 2],
                                               optimizer=optimizer)
    slu_f1_scores = [] #early stop
    for epoch in range(config.epochs):
        model.train()
        losses_slu = []
        losses_slm = []
        losses_all = []
        scheduler.step()
        for i, (batch_1, batch_2) in enumerate(zip(data_loader(train_data_1, config.batch_size, True),
                                                   data_loader(train_data_2, config.batch_size, True))):
            h, c, slot, intent = pad_to_batch(batch_1, model.vocab, model.slot_vocab)
            h = [hh.to(device) for hh in h]
            c = c.to(device)
            slot = slot.to(device)
            intent = intent.to(device)

            slot_p, intent_p = model(h, c)
            loss_s = slot_loss_function(slot_p, slot.view(-1))
            loss_i = intent_loss_function(intent_p, intent.view(-1))
            loss_slu = loss_s + loss_i
            losses_slu.append(loss_slu.item())
            if config.slm_weight>0:
                slm_h, slm_candi, slm_label = pad_to_batch_slm(batch_2, model.vocab)
                slm_h = [hh.to(device) for hh in slm_h]
                slm_candi = [hh.to(device) for hh in slm_candi]
                slm_label = slm_label.to(device)
                slm_p = model(slm_h, slm_candi, slm=True).view(-1, 2)

                loss_slm = slm_loss(slm_p, slm_label.view(-1))
                losses_slm.append(loss_slm.item())
            else:
                loss_slm = 0
                losses_slm.append(loss_slm)

            optimizer.zero_grad()
            # if epoch >= config.epochs//4:
            #     config.slm_weight = config.slm_weight/2
            # elif epoch >= config.epochs//2:
            #config.slm_weight = config.slm_weight/(epoch+1)
            loss = loss_slm * config.slm_weight + (1 - config.slm_weight) * loss_slu
            losses_all.append(loss.item())

            loss.backward()
            optimizer.step()

            if i % 40 == 0:
                # SLU
                intent_acc = accuracy_score(intent.view(-1).tolist(), intent_p.max(1)[1].tolist())
                slot_f1 = f1_score(slot.view(-1).tolist(), slot_p.max(1)[1].tolist(), average='micro')
                # SLM
                if config.slm_weight > 0:
                    label = slm_label.view(-1).tolist()
                    pred = slm_p.max(1)[1].tolist()
                    slm_acc = accuracy_score(label, pred)
                    slm_recall = recall_score(label, pred)
                else:
                    slm_acc = 0
                    slm_recall = 0
                metrics_dict = {'loss_all': np.round(np.mean(losses_all),2),
                                'loss_slm': np.round(np.mean(losses_slm),2),
                                'losses_slu': np.round(np.mean(losses_slu),2),
                                # 'intent_acc': np.round(intent_acc,2),
                                # 'slot_f1': np.round(slot_f1,2),
                                # 'slm_acc': np.round(slm_acc,2),
                                # 'slm_recall': np.round(slm_recall,2)
                                }
                log_printer(log, "train", epoch="{}/{}".format(epoch, config.epochs),
                            iters="{}/{}".format(i, len(train_data_1) // config.batch_size),
                            metrics=metrics_dict)
                losses_all = []
                losses_slu = []
                losses_slm = []

        metric, loss = evaluation_multi(model, dev_data_1, dev_data_2, config)

        metrics_dict = {'loss_all': np.round(np.mean(loss[0]),2),
                        'loss_slm':  np.round(np.mean(loss[1]),2),
                        'losses_slu':  np.round(np.mean(loss[2]),2),
                        'intent_acc':  np.round(metric[0],2),
                        'slot_f1':  np.round(metric[1],2),
                        'slm_acc':  np.round(metric[2],2),
                        'slm_recall':  np.round(metric[3],2)
                        }
        log_printer(log, 'eval', epoch="{}/{}".format(epoch, config.epochs),
                    iters="{}/{}".format(i, len(train_data_1) // config.batch_size),
                    metrics=metrics_dict)
        early_metric = -loss[2]#metric[1]
        if early_metric > config.best_score:
            slu_f1_scores = []
            config.best_score = early_metric
            evaluation(model, (test_data_1,test_data_2),config)
            save(model, config)
        slu_f1_scores.append(early_metric)
        if len(slu_f1_scores) > config.early_stop and config.early_stop!=0:
            print('Early stop after f1 score did not increase after {} epochs'.format(config.early_stop))
            return

def evaluation_multi(model, dev_data_1, dev_data_2,config):
    model.eval()
    slm_loss = nn.CrossEntropyLoss()
    slot_loss_function = nn.CrossEntropyLoss(ignore_index=0)
    intent_loss_function = nn.CrossEntropyLoss()

    intent_label = []
    intent_pred = []
    slot_label = []
    slot_pred = []

    losses_slu = []
    losses_slm = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader(dev_data_1, 32, False)):
            h, c, slot, intent = pad_to_batch(batch, model.vocab, model.slot_vocab)
            h = [hh.to(device) for hh in h]
            c = c.to(device)
            slot = slot.to(device)
            intent = intent.to(device)
            slot_p, intent_p = model(h, c)

            intent_label.extend(intent.view(-1).tolist())
            intent_pred.extend(intent_p.max(1)[1].tolist())

            slot_label.extend(slot.view(-1).tolist())
            slot_pred.extend(slot_p.max(1)[1].tolist())

            loss_s = slot_loss_function(slot_p, slot.view(-1))
            loss_i = intent_loss_function(intent_p, intent.view(-1))
            losses_slu.append((loss_s.item() + loss_i.item()))
        if config.slm_weight > 0:
            slm_label_all = []
            slm_pred_all = []
            for i, batch in enumerate(data_loader(dev_data_2, 32, False)):
                slm_h, slm_candi, slm_label = pad_to_batch_slm(batch, model.vocab)
                slm_h = [hh.to(device) for hh in slm_h]
                slm_candi = [hh.to(device) for hh in slm_candi]
                slm_label = slm_label.to(device)
                slm_p = model(slm_h, slm_candi, slm=True).view(-1, 2)
                slm_label_all.extend(slm_label.view(-1).tolist())
                slm_pred_all.extend(slm_p.max(1)[1].tolist())
                loss_slm = slm_loss(slm_p, slm_label.view(-1))
                losses_slm.append(loss_slm.item())
            slm_acc = accuracy_score(slm_label_all,slm_pred_all)
            slm_recall = recall_score(slm_label_all, slm_pred_all)
        else:
            losses_slm.append(0)
            slm_acc = 0
            slm_recall = 0
    slot_to_index={k: v for k, v in model.slot_vocab.items()}
    sorted_labels = list(set(slot_label) - {slot_to_index['O'], slot_to_index['<pad>']})

    intent_acc = accuracy_score(intent_label, intent_pred)
    slot_f1 = f1_score(slot_label,slot_pred,average='micro',labels=sorted_labels)
    
    preds = [[y] for y in slot_pred]
    labels = [[y] for y in slot_label]

    slot_f1 = float(metrics.flat_classification_report(
        labels, preds, labels=sorted_labels, digits=3).split('\n')[-2].split()[-2])

    losses_slm = np.mean(losses_slm)
    losses_slu = np.mean(losses_slu)
    losses_all = losses_slu + losses_slm
    return [intent_acc, slot_f1, slm_acc, slm_recall], [losses_all, losses_slm,losses_slu]

def evaluation(model, dev_data, config):
    model.eval()
    dev_data_1, dev_data_2 = dev_data
    index2slot = {v: k for k, v in model.slot_vocab.items()}
    preds = []
    labels = []
    hits = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader(dev_data_1, 32, False)):
            h, c, slot, intent = pad_to_batch(batch, model.vocab, model.slot_vocab)
            h = [hh.to(device) for hh in h]
            c = c.to(device)
            slot = slot.to(device)
            intent = intent.to(device)
            slot_p, intent_p = model(h, c)

            preds.extend([index2slot[i] for i in slot_p.max(1)[1].tolist()])
            labels.extend([index2slot[i] for i in slot.view(-1).tolist()])
            hits += torch.eq(intent_p.max(1)[1], intent.view(-1)).sum().item()
        if config.slm_weight>0:
            slm_label_all = []
            slm_pred_all = []
            for i, batch in enumerate(data_loader(dev_data_2, 32, False)):
                slm_h, slm_candi, slm_label = pad_to_batch_slm(batch, model.vocab)
                slm_h = [hh.to(device) for hh in slm_h]
                slm_candi = [hh.to(device) for hh in slm_candi]
                slm_label = slm_label.to(device)
                slm_p = model(slm_h, slm_candi, slm=True).view(-1, 2)
                slm_label_all.extend(slm_label.view(-1).tolist())
                slm_pred_all.extend(slm_p.max(1)[1].tolist())

            slm_acc = accuracy_score(slm_label_all, slm_pred_all)
            slm_recall = recall_score(slm_label_all, slm_pred_all)
            print('slm accuracy:\t%.5f' % slm_acc)
            print('slm recall:\t%.5f' % slm_recall)
    intent_acc = hits / len(dev_data_1)
    print('intent accuracy:\t%.5f' % intent_acc)

    sorted_labels = sorted(
        list(set(labels) - {'O', '<pad>'}),
        key=lambda name: (name[1:], name[0])
    )

    # this is because sklearn_crfsuite.metrics function flatten inputs
    preds = [[y] for y in preds]
    labels = [[y] for y in labels]

    print(metrics.flat_classification_report(
        labels, preds, labels=sorted_labels, digits=3
    ))

def inference(model, test_data):
    model.eval()
    index2slot = {v: k for k, v in model.slot_vocab.items()}
    index2word = {v: k for k, v in model.vocab.items()}
    index2intent = {v: k for k, v in model.intent_vocab.items()}

    f_temp = open('result.txt','w',encoding='utf-8')
    with torch.no_grad():
        for i, batch in enumerate(data_loader(test_data, 32, False)):
            h, c, slot, intent = pad_to_batch(batch, model.vocab, model.slot_vocab)
            h = [hh.to(device) for hh in h]
            c = c.to(device)
            slot = slot.to(device)
            intent = intent.to(device)
            slot_p, intent_p = model(h, c)

            preds = [index2slot[i] for i in slot_p.max(1)[1].tolist()]
            labels = [index2slot[i] for i in slot.view(-1).tolist()]
            words = [index2word[i] for i in c.view(-1).tolist()]
            intents_pred = [index2intent[i] for i in intent_p.max(1)[1].tolist()]
            intents_label =[index2intent[i] for i in intent.view(-1).tolist()]

            length = c.size(1)

            for ip, il in zip(intents_pred, intents_label):
                f_temp.write(ip+'\t'+il+'\n')
                i = 0
                while words[0] =='<pad>':
                    preds.pop(0)
                    labels.pop(0)
                    words.pop(0)

                while words!=[] and words[0]!='<pad>' and i<length:
                    i+=1
                    f_temp.write(words[0]+'\t'+preds[0]+'\t'+labels[0]+'\n')
                    preds.pop(0)
                    labels.pop(0)
                    words.pop(0)
                f_temp.write('\n')



def model_init(built_vocab, config):

    word2index, slot2index, intent2index = built_vocab
    model = model_dic[config.model](len(word2index),config.embed_size,config.hidden_size,\
                 len(slot2index),len(intent2index),word2index['<pad>'])
    model.to(device)

    model.vocab = word2index
    model.slot_vocab = slot2index
    model.intent_vocab = intent2index
    config.best_score = 0

    return model

def model_load(config):
    print('loading previous model...')
    checkpoint = torch.load(config.save_path + '/model.pkl', map_location=lambda storage, loc: storage)
    print(checkpoint['config'])
    word2index, slot2index, intent2index = checkpoint['vocab'], checkpoint['slot_vocab'], checkpoint['intent_vocab']
    model = model_dic[config.model](len(word2index), config.embed_size, config.hidden_size, \
                                    len(slot2index), len(intent2index), word2index['<pad>'])
    model.to(device)
    model.vocab = word2index
    model.slot_vocab = slot2index
    model.intent_vocab = intent2index

    model.load_state_dict(checkpoint['model'])
    config.best_score = checkpoint['best_score']
    #model = torch.load(os.path.join(config.save_path,'model.pkl'), map_location=lambda storage, loc: storage)
    return model

def save(model,config):
    checkpoint = {
                'model': model.state_dict(),
                'vocab': model.vocab,
                'slot_vocab': model.slot_vocab,
                'intent_vocab': model.intent_vocab,
                'config': config,
                'best_score': config.best_score
            }
    torch.save(checkpoint,os.path.join(config.save_path,'model.pkl'))
    #torch.save(model, os.path.join(config.save_path,'model.pkl'))
    print("Model saved!")


def log_printer(log, name, epoch, iters, metrics):
    metrics_dict = metrics
    step = int(iters.split('/')[0]) + int(iters.split('/')[1]) * (int(epoch.split('/')[0]) - 1)
    print("[ {} epoch:{} iter:{} ]".format(name, epoch, iters) + str(metrics_dict))

    for k,v in metrics_dict.items():
        if name == 'train':
            k = 'valid_'+k
        log.scalar_summary(tag=k, value=v, step=step)


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',type=str, default='kvret', help='dataset selection of kvret or m2m')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epochs', type=int, default=30,
                        help='num_epochs')
    parser.add_argument('--pre_dataset', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning_rate')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout')
    parser.add_argument('--embed_size', type=int, default=100,
                        help='embed_size')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='hidden_size')
    parser.add_argument('--save_path', type=str, default='weight',
                        help='save_path')
    parser.add_argument('--model', type=str, default='sden',
                        help='s2s, contex_s2s, sden' )
    parser.add_argument('--slm_weight',type=float, default=0,
                        help='slm weight')
    parser.add_argument('--model_name',type=str, default='sden_slm0',
                        help='name of modelfile')
    parser.add_argument('--new_model',action='store_true',
                         help='whether delete previous model or not')
    parser.add_argument('--multi_domain',action='store_true',
                         help='whether use multi-domain dataset or not')
    parser.add_argument('--early_stop', type=int, default=5,
                        help='whether delete previous model or not')
    config = parser.parse_args()
    config.save_path = os.path.join(config.save_path , config.task , config.model_name)

    options = vars(config)
    for k,v in options.items():
        print('[  {}:  {}  ]'.format(k,v))
    return config
