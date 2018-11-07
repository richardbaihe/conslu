from utils.data_utils import *
from utils.train_utils import *

if __name__ == "__main__":
    config = get_config()
    if config.pre_dataset:
        print('convert json to iob for kvret')
        json2iob_kvret()
        print('convert json to iob for m2m')
        json2iob_m2m()

    train_path = os.path.join('data',config.task,'train.iob')
    dev_path = os.path.join('data',config.task,'dev.iob')
    test_path = os.path.join('data',config.task,'test.iob')
    train_data, train_slm_data, built_vocab = prepare_dataset(train_path,config)
    dev_data, dev_slm_data = prepare_dataset(dev_path,config,built_vocab)
    test_data, test_slm_data = prepare_dataset(test_path,config,built_vocab)

    if os.path.exists(config.save_path + '/model.pkl'):
        print('\n[  found previous model from {}  ]'.format(config.save_path))
        if config.new_model and config.mode == 'train':
            print('deleting previous model...')
            os.system('rm -rf ' + config.save_path)
            print('initializing new model...')
            model = model_init(built_vocab, config)
        else:
            model = model_load(config)
    else:
        print('no previous model, initializing new model...')
        model = model_init(built_vocab, config)

    if config.mode == 'train':
        print('begin training...')
        train_multitask(model,(train_data,train_slm_data),(dev_data,dev_slm_data),config)
        print('begin testing...')
        model = model_load(config)
        metric, loss = evaluation_multi(model,dev_data,dev_slm_data,config)
        metrics_dict = {'loss_all': np.round(np.mean(loss[0]),2),
                        'loss_slm':  np.round(np.mean(loss[1]),2),
                        'losses_slu':  np.round(np.mean(loss[2]),2),
                        'intent_acc':  np.round(metric[0],2),
                        'slot_f1':  np.round(metric[1],2),
                        'slm_acc':  np.round(metric[2],2),
                        'slm_recall':  np.round(metric[3],2)
                        }
        print(str(metrics_dict))
        evaluation(model, test_data)
    else:
        metric, loss = evaluation_multi(model,dev_data,dev_slm_data,config)
        metrics_dict = {'loss_all': np.round(np.mean(loss[0]),2),
                        'loss_slm':  np.round(np.mean(loss[1]),2),
                        'losses_slu':  np.round(np.mean(loss[2]),2),
                        'intent_acc':  np.round(metric[0],2),
                        'slot_f1':  np.round(metric[1],2),
                        'slm_acc':  np.round(metric[2],2),
                        'slm_recall':  np.round(metric[3],2)
                        }
        print(str(metrics_dict))
        evaluation(model, test_data)
