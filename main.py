import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data_utils import *
from model import SDEN
from sklearn_crfsuite import metrics
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_data, config):
    slot_loss_function = nn.CrossEntropyLoss(ignore_index=0)
    intent_loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1, milestones=[config.epochs // 4, config.epochs // 2],
                                               optimizer=optimizer)

    model.train()
    for epoch in range(config.epochs):
        losses = []
        scheduler.step()
        for i, batch in enumerate(data_loader(train_data, config.batch_size, True)):
            h, c, slot, intent = pad_to_batch(batch, model.vocab, model.slot_vocab)
            h = [hh.to(device) for hh in h]
            c = c.to(device)
            slot = slot.to(device)
            intent = intent.to(device)
            model.zero_grad()
            slot_p, intent_p = model(h, c)

            loss_s = slot_loss_function(slot_p, slot.view(-1))
            loss_i = intent_loss_function(intent_p, intent.view(-1))
            loss = loss_s + loss_i
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("[%d/%d] [%d/%d] mean_loss : %.3f" % \
                      (epoch, config.epochs, i, len(train_data) // config.batch_size, np.mean(losses)))
                losses = []

def train_multitask(model,train_data,dev_data,config):
    train_data_1, train_data_2 = train_data
    dev_data_1, dev_data_2 = dev_data
    slm_loss = nn.CrossEntropyLoss()
    slot_loss_function = nn.CrossEntropyLoss(ignore_index=0)
    intent_loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1, milestones=[config.epochs // 4, config.epochs // 2],
                                               optimizer=optimizer)

    model.train()
    for epoch in range(config.epochs):
        losses = []
        scheduler.step()
        for i, (batch_1,batch_2) in enumerate(zip(data_loader(train_data_1, config.batch_size, True),
                                                  data_loader(train_data_2, config.batch_size, True))):
            h, c, slot, intent = pad_to_batch(batch_1, model.vocab, model.slot_vocab)
            h = [hh.to(device) for hh in h]
            c = c.to(device)
            slot = slot.to(device)
            intent = intent.to(device)

            slm_h, slm_candi, slm_label = pad_to_batch_slm(batch_2, model.vocab)
            slm_h = [hh.to(device) for hh in slm_h]
            slm_candi = [hh.to(device) for hh in slm_candi]
            slm_label = slm_label.to(device)

            model.zero_grad()
            slot_p, intent_p = model(h, c)
            slm_p = model(slm_h,slm_candi,slm=True)

            loss_s = slot_loss_function(slot_p, slot.view(-1))
            loss_i = intent_loss_function(intent_p, intent.view(-1))
            loss_slm = slm_loss(slm_p.view(-1,2),slm_label.view(-1))
            loss = loss_s + loss_i + loss_slm
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("[%d/%d] [%d/%d] mean_loss : %.3f" % \
                      (epoch, config.epochs, i, len(train_data) // config.batch_size, np.mean(losses)))
                losses = []
            evaluation_slm(model,dev_slm_data)
            evaluation(model,dev_data)
def evaluation_slm(model,dev_data):
    model.eval()
    pos = 0
    hits = 0
    total = 0
    total_pos = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader(dev_data, 32, True)):
            slm_h, slm_candi, slm_label = pad_to_batch_slm(batch, model.vocab)
            slm_h = [hh.to(device) for hh in slm_h]
            slm_candi = [hh.to(device) for hh in slm_candi]
            slm_label = slm_label.to(device)
            slm_p = model(slm_h, slm_candi, slm=True)
            correct = torch.eq(slm_p.max(1)[1], slm_label.view(-1))
            hits += correct.sum().item()
            total += slm_label.shape[0]
            pos += torch.eq(correct,slm_label.view(-1)).sum().item()
            total_pos += slm_label.sum().item()
    print('slm accuracy:\t%.5f' % hits/total)
    print('slm recall:\t%.5f' % pos/total_pos)


def evaluation(model,dev_data):
    model.eval()
    index2slot = {v:k for k,v in model.slot_vocab.items()}
    preds=[]
    labels=[]
    hits=0
    with torch.no_grad():
        for i,batch in enumerate(data_loader(dev_data,32,True)):
            h,c,slot,intent = pad_to_batch(batch,model.vocab,model.slot_vocab)
            h = [hh.to(device) for hh in h]
            c = c.to(device)
            slot = slot.to(device)
            intent = intent.to(device)
            slot_p, intent_p = model(h,c)

            preds.extend([index2slot[i] for i in slot_p.max(1)[1].tolist()])
            labels.extend([index2slot[i] for i in slot.view(-1).tolist()])
            hits+=torch.eq(intent_p.max(1)[1],intent.view(-1)).sum().item()


    print('intent accuracy:\t%.5f' % hits/len(dev_data))
    
    sorted_labels = sorted(
    list(set(labels) - {'O','<pad>'}),
    key=lambda name: (name[1:], name[0])
    )
    
    # this is because sklearn_crfsuite.metrics function flatten inputs
    preds = [[y] for y in preds] 
    labels = [[y] for y in labels]
    
    print(metrics.flat_classification_report(
    labels, preds, labels = sorted_labels, digits=3
    ))

def save(model,config):
    checkpoint = {
                'model': model.state_dict(),
                'vocab': model.vocab,
                'slot_vocab' : model.slot_vocab,
                'intent_vocab' : model.intent_vocab,
                'config' : config,
            }
    torch.save(checkpoint,config.save_path)
    print("Model saved!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=5,
                        help='num_epochs')
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
    parser.add_argument('--save_path', type=str, default='weight/model.pkl',
                        help='save_path')
    parser.add_argument('--model', type=str, default='sden',
                        help='seq2seq, memory, sden' )
    parser.add_argument('--slm',type=bool, default=False,
                        help='whether sentence level language model training or not')
    config = parser.parse_args()
    
    train_data, train_slm_data, word2index, slot2index, intent2index = prepare_dataset('data/train.iob',slm=config.slm)
    dev_data, dev_slm_data = prepare_dataset('data/dev.iob',(word2index,slot2index,intent2index),slm=config.slm)
    if config.model == 'sden':
        model = SDEN(len(word2index),config.embed_size,config.hidden_size,\
                     len(slot2index),len(intent2index),word2index['<pad>'])
    model.to(device)
    model.vocab = word2index
    model.slot_vocab = slot2index
    model.intent_vocab = intent2index

    if config.mode == 'train':
        if config.slm:
            train_multitask(model,(train_data,train_slm_data),(dev_data,dev_slm_data),config)
        else:
            train(model, train_data,dev_data, config)
        save(model, config)
    evaluation(model, dev_data)