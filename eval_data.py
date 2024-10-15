
import numpy as np
import pandas as pd
import torch
import time
import math
from module import TRTKC
from nx2graphs import load_real_data, load_train_real_data, load_real_true_TKC


class MLP(torch.nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 64)
        self.fc_2 = torch.nn.Linear(64, 32)
        self.fc_3 = torch.nn.Linear(32, 1)

        self.act = torch.nn.ReLU()

        torch.nn.init.kaiming_normal_(self.fc_1.weight)
        torch.nn.init.kaiming_normal_(self.fc_2.weight)
        torch.nn.init.kaiming_normal_(self.fc_3.weight)

        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)



DATA='slashdot-threads'
device = torch.device('cuda:{}'.format('0'))
n_feat = np.load('./data/test/Real/processed/ml_{}_node.npy'.format(DATA), allow_pickle=True)
# test_real_feat = np.load('./data/test/Real/processed/ml_{}_node.npy'.format(DATA), allow_pickle=True)
test_real_feat = np.zeros((4200000, 128))

MLP_model = MLP(n_feat.shape[1], drop=0.1)
MLP_model = MLP_model.to(device)

train_real_src_l, train_real_dst_l, train_real_ts_l, train_real_node_count, train_real_node, train_real_time, \
    train_real_ngh_finder = load_train_real_data('store_true')

tatkc = TRTKC(train_real_ngh_finder[0], test_real_feat, num_layers=2, use_time='time',
              agg_method='lstm', attn_mode='prod', seq_len=20, n_head=2, drop_out=0.1)
tatkc = tatkc.to(device)

optimizer = torch.optim.Adam(list(tatkc.parameters()) + list(MLP_model.parameters()), 0.01)

test_real_src_l, test_real_dst_l, test_real_ts_l, test_real_node_count, test_real_node, test_real_time, \
    test_real_ngh_finder = load_real_data(dataName=DATA)
nodeList_test_real, test_label_l_real = load_real_true_TKC('{}'.format(DATA))

tatkc.load_state_dict(torch.load(f'./saved_models/model_tatkc_LSTM.pth'))
MLP_model.load_state_dict(torch.load(f'./saved_models/model_MLP_LSTM.pth'))

real_data_start_time = time.time()
test_pred_tbc_list = []
tatkc.ngh_finder=test_real_ngh_finder
BATCH_SIZE=1500
src = nodeList_test_real
test_real_ts_list = np.array([test_real_time] * len(nodeList_test_real))
ts=test_real_ts_list
label =test_label_l_real
with torch.no_grad():
    MLP_model=MLP_model.eval()
    TEST_BATCH_SIZE = BATCH_SIZE
    num_test_instance = len(src)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
    for k in range(num_test_batch):
        s_idx = k * TEST_BATCH_SIZE
        e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
        test_src_l_cut = np.array(src[s_idx:e_idx])
        test_ts_l_cut = np.array(ts[s_idx:e_idx])
        src_embed = tatkc.tem_conv(test_src_l_cut, test_ts_l_cut, 2, num_neighbors=14)
        test_pred_tbc = MLP_model(src_embed)
        test_pred_tbc_list.extend(test_pred_tbc.cpu().detach().numpy().tolist())
    real_data_end_time = time.time()
    # print(test_pred_tbc_list)
    print(len(test_pred_tbc_list))
    print(len(label))
    data = {
        'predicted_tkc': test_pred_tbc_list,
        'true_tkc': label
    }
    # 将字典转换为 Pandas DataFrame
    df = pd.DataFrame(data)

    # 将 DataFrame 保存为 CSV 文件
    df.to_csv(f'Result_{DATA}.csv', index=False)

    predicted_tensor= torch.tensor(test_pred_tbc_list, dtype=torch.float32)
    label_tensor = torch.tensor(label, dtype=torch.float32)
    MAE = torch.nn.L1Loss()
    MSE = torch.nn.MSELoss()
    mae_v = MAE(predicted_tensor, label_tensor)
    mse_v = MSE(predicted_tensor, label_tensor)
    print(f'MAE: {mae_v.item()}, MSE: {mse_v.item()}, RMSE: {np.sqrt(mse_v.item())}')
    print(real_data_end_time-real_data_start_time)
