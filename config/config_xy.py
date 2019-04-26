# coding = utf-8
# author = xy


class Config:
    name = 'ModelXy'
    mode = 'LSTM'
    hidden_size = 64
    dropout_p = 0.2
    encoder_layer_num = 2

    batch_size = 64
    test_batch_size = 64
    epochs = 50
    early_stop = 5
    lr = 1e-3

    train_path = '../data/train_data.json'
    val_path = '../data/dev_data.json'

config = Config()
