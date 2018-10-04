import torch.optim as optim
import numpy as np
import logger
from data_utils import *
from model import SDEN
from sklearn_crfsuite import metrics
from sklearn.metrics import f1_score,accuracy_score,recall_score
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

    for epoch in range(config.epochs):
        model.train()
        losses_slu = []
        losses_slm = []
        losses_all = []
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
            slm_p = model(slm_h,slm_candi,slm=True).view(-1,2)

            loss_s = slot_loss_function(slot_p, slot.view(-1))
            loss_i = intent_loss_function(intent_p, intent.view(-1))
            loss_slm = slm_loss(slm_p,slm_label.view(-1))
            loss = loss_s + loss_i + loss_slm
            losses_slm.append(loss_slm.item())
            losses_slu.append((loss_s + loss_i).item())
            losses_all.append(loss.item())
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                #SLU
                hits = torch.eq(intent_p.max(1)[1], intent.view(-1)).sum().item()
                intent_acc = hits / config.batch_size
                slot_f1 = f1_score(slot.view(-1).tolist(),slot_p.max(1)[1].tolist(),average='micro')
                #SLM
                label = slm_label.view(-1).tolist()
                pred = slm_p.max(1)[1].tolist()
                slm_acc = accuracy_score(label,pred)
                slm_recall = recall_score(label,pred)

                log_printer("train", epoch="{}/{}".format(epoch, config.epochs),
                            iters="{}/{}".format(i, len(train_data_1) // config.batch_size),
                            metrics=[intent_acc,slot_f1,slm_acc,slm_recall],
                            loss=[np.mean(losses_all),np.mean(losses_slm),np.mean(losses_slu)])
                losses_all = []
                losses_slu = []
                losses_slm = []
        metric, loss = evaluation_multi(model, dev_data_1, dev_data_2)
        log_printer('eval', epoch="{}/{}".format(epoch, config.epochs),
                    iters="{}/{}".format(i, len(train_data_1) // config.batch_size),
                    metrics=metric,
                    loss=loss)
    evaluation(model, dev_data_1)

def evaluation_multi(model,dev_data_1,dev_data_2):
    model.eval()
    slm_loss = nn.CrossEntropyLoss()
    slot_loss_function = nn.CrossEntropyLoss(ignore_index=0)
    intent_loss_function = nn.CrossEntropyLoss()
    intent_acc = []
    slot_f1 = []
    slm_acc = []
    slm_recall = []
    losses_slu = []
    losses_slm = []
    with torch.no_grad():
        for i,batch in enumerate(data_loader(dev_data_1,32,True)):
            h,c,slot,intent = pad_to_batch(batch,model.vocab,model.slot_vocab)
            h = [hh.to(device) for hh in h]
            c = c.to(device)
            slot = slot.to(device)
            intent = intent.to(device)
            slot_p, intent_p = model(h,c)

            hits = torch.eq(intent_p.max(1)[1], intent.view(-1)).sum().item()
            intent_acc.append(hits / config.batch_size)
            slot_f1 = f1_score(slot.view(-1).tolist(), slot_p.max(1)[1].tolist(), average='micro')
            loss_s = slot_loss_function(slot_p, slot.view(-1))
            loss_i = intent_loss_function(intent_p, intent.view(-1))
            losses_slu.append((loss_s.item()+loss_i.item()))
        for i, batch in enumerate(data_loader(dev_data_2, 32, True)):
            slm_h, slm_candi, slm_label = pad_to_batch_slm(batch, model.vocab)
            slm_h = [hh.to(device) for hh in slm_h]
            slm_candi = [hh.to(device) for hh in slm_candi]
            slm_label = slm_label.to(device)
            slm_p = model(slm_h, slm_candi, slm=True).view(-1,2)
            label = slm_label.view(-1).tolist()
            pred = slm_p.max(1)[1].tolist()
            slm_acc = accuracy_score(label, pred)
            slm_recall = recall_score(label, pred)
            loss_slm = slm_loss(slm_p,slm_label.view(-1))
            losses_slm.append(loss_slm.item())

    losses_slm = np.mean(losses_slm)
    losses_slu = np.mean(losses_slu)
    losses_all = losses_slu + losses_slm
    return [np.mean(intent_acc),np.mean(slot_f1),np.mean(slm_acc),np.mean(slm_recall)],[losses_all, losses_slm, losses_slu]

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

    intent_acc = hits / len(dev_data)
    print('intent accuracy:\t%.5f' % intent_acc)
    
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

def log_printer(name, metrics, loss, epoch=None, iters=None):
    if name == 'train':
        print("{}\tepoch : {}\titer : {}\tintent_acc : {.3f}\tslot_f1 : {.3f}\tslm_acc : {.3f}\tslm_r : {.3f}\tloss_all : {.3f}\tloss_slm : {.3f}\tloss_slu : {.3f}".format(
            name, epoch, iters, metrics[0], metrics[1], metrics[2], metrics[3], loss[0], loss[1], loss[2]
        ))
        step = int(iters.split('/')[0]) + int(iters.split('/')[1]) * (int(epoch.split('/')[0])-1)
        log.scalar_summary(tag="loss_all", value=loss[0], step=step)
        log.scalar_summary(tag="loss_slm", value=loss[1], step=step)
        log.scalar_summary(tag="loss_slu", value=loss[2], step=step)

    else:
        if loss == None:
            print("{}\tintent_acc : {.3f}\tslot_f1 : {.3f}\tslm_acc : {.3f}\tslm_r : {.3f}".format(
                name, metrics[0], metrics[1], metrics[2], metrics[3]))
        else:
            print("{}\tintent_acc : {.3f}\tslot_f1 : {.3f}\tslm_acc : {.3f}\tslm_r : {.3f}\tloss_all : {.3f}\tloss_slm : {.3f}\tloss_slu : {.3f}".format(
                name, metrics[0], metrics[1], metrics[2], metrics[3], loss[0], loss[1], loss[2]))
        if iters != None and epoch != None and loss != None:
            step = int(iters.split('/')[0]) + int(iters.split('/')[1]) * (int(epoch.split('/')[0])-1)
            log.scalar_summary(tag="valid_loss_all", value=loss[0], step=step)
            log.scalar_summary(tag="valid_loss_slm", value=loss[1], step=step)
            log.scalar_summary(tag="valid_loss_slu", value=loss[2], step=step)

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
    parser.add_argument('--tensorboard',type=str, default='logs',
                        help='path for logs')
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

    log = logger.Logger(config.tensorboard)

    if config.mode == 'train':
        if config.slm:
            train_multitask(model,(train_data,train_slm_data),(dev_data,dev_slm_data),config)
        else:
            train(model, train_data, config)
        save(model, config)
    evaluation(model, dev_data)