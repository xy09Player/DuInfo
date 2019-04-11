# coding = utf-8
# author = xy


class Config:
    name = 'ModelSpo'
    mode = 'LSTM'
    hidden_size = 64
    dropout_p = 0.2
    encoder_layer_num = 2

    batch_size = 64
    test_batch_size = 64
    epochs = 50
    early_stop = 5
    lr = 1e-3

    model_path = 'model_spo_1'
    model_path_sbj = 'model_sbj_1'

config = Config()
