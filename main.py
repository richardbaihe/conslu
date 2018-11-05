from utils.data_utils import *
from utils.train_utils import *
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    config = get_config()

    train_path = os.path.join('data',config.task,'train.iob')
    dev_path = os.path.join('data',config.task,'dev.iob')
    test_path = os.path.join('data',config.task,'test.iob')
    train_data, train_slm_data, built_vocab = prepare_dataset(train_path,config)
    dev_data, dev_slm_data = prepare_dataset(dev_path,config,built_vocab)
    test_data, test_slm_data = prepare_dataset(test_path,config,built_vocab)

    model = model_init(built_vocab, config)

    if config.mode == 'train':
        print('begin training...')
        train_multitask(model,(train_data,train_slm_data),(dev_data,dev_slm_data),config)
        print('begin testing...')
        evaluation(model, test_data)
    else:
        evaluation(model, test_data)
