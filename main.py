from utils.data_utils import *
from utils.train_utils import *
import sys

def json2iob():
    print('convert json to iob for kvret')
    json2iob_kvret()
    print('convert json to iob for m2m')
    json2iob_m2m()


def train(config):
    print('begin training...')
    BERT = 'bert' in config.model_name
    if config.new_model:
        vocab = build_vocab(config.train_path,bert=BERT)
        print('initializing new model...')
        model = model_init(vocab, config)
    elif os.path.exists(config.save_path + '/model.pkl'):
        model = model_load(config)
        vocab = [model.vocab, model.slot_vocab, model.intent_vocab]
    else:
        print('no model file found in {}'.format(config.save_path + '/model.pkl'))
        sys.exit()
    if BERT:
        train_data, train_slm_data = prepare_dataset_bert(config.train_path, config, vocab)
        dev_data, dev_slm_data = prepare_dataset_bert(config.dev_path, config, vocab)
        test_data, test_slm_data = prepare_dataset_bert(config.test_path, config, vocab)
    else:
        train_data, train_slm_data = prepare_dataset(config.train_path, config, vocab)
        dev_data, dev_slm_data = prepare_dataset(config.dev_path, config, vocab)
        test_data, test_slm_data = prepare_dataset(config.test_path, config, vocab)

    train_multitask(model, (train_data, train_slm_data), (dev_data, dev_slm_data), config, (test_data, test_slm_data))
    print('begin testing...')
    test(model, (test_data,test_slm_data),config)

def test(config):
    print('begin testing...')
    model = model_load(config)
    vocab = [model.vocab, model.slot_vocab, model.intent_vocab]
    test_data, test_slm_data = prepare_dataset(config.test_path, config, vocab)
    test(model, (test_data,test_slm_data),config)

if __name__ == "__main__":
    config = get_config()
    config.train_path = os.path.join('data',config.task,'train.iob')
    config.dev_path = os.path.join('data',config.task,'dev.iob')
    config.test_path = os.path.join('data',config.task,'test.iob')

    if config.pre_dataset:
        json2iob()
        kvret2kvret_star(config)
    if config.mode == 'train':
        train(config)
    else:
        test(config)
