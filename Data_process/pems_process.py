import numpy as np
import pandas as pd

dir = '/root/Lane_level/AGWN/Data_process/data/'


def concat_slice():
    """将单个sensors的切片合成一个"""
    slice = ['one', 'two', 'three', 'four']
    for i in range (1, 5):
        speed_df = pd.DataFrame ()
        for j in slice:
            suffix = 'sensors%s/%s.csv' % (i, j)
            single_slice = pd.read_csv (dir + suffix , index_col=0)
            speed_df = pd.concat ([speed_df, single_slice], axis=0)
            print (speed_df.shape)
        speed_df = speed_df.loc['2/5/2017 0:00':'3/4/2017 23:55', :]
        speed_df.to_csv (dir + 'sensors%s/speedflow.csv' % i)

def Split():
    """切分速度和流量"""
    for i in range (1, 5):
        suffix = 'sensors%s/speedflow.csv' % i
        speedflow = pd.read_csv (dir + suffix, index_col=0)
        col = ['sensors %s Lane 1' % i, 'sensors %s Lane 2' % i, 'sensors %s Lane 3' % i, 'sensors %s Lane 4' % i,
               'sensors %s Lane 5' % i]
        speed = speedflow.iloc[:, 0:5]
        speed.columns = col
        speed = speed.fillna(method='ffill',axis=0).fillna(method='bfill',axis=0)
        flow = speedflow.iloc[:, 5:10]
        flow.columns = col
        flow[flow < 1] = None
        flow = flow.fillna(method='ffill' ,axis=0).fillna(method='bfill',axis=0)
        speed.to_csv (dir + 'sensors%s/speed.csv' % i)
        flow.to_csv (dir + 'sensors%s/flow.csv' % i)

def fill_sensors4():
    speed = pd.read_csv (dir + 'sensors4/speed.csv', index_col=0)
    flow = pd.read_csv (dir + 'sensors4/flow.csv', index_col=0)
    speed['sensors 4 Lane 5'] = speed[
        ['sensors 4 Lane 1', 'sensors 4 Lane 2', 'sensors 4 Lane 3', 'sensors 4 Lane 4']].mean (axis=1)
    flow['sensors 4 Lane 5'] = flow[
        ['sensors 4 Lane 1', 'sensors 4 Lane 2', 'sensors 4 Lane 3', 'sensors 4 Lane 4']].mean (axis=1)
    speed.to_csv (dir + 'sensors4/speed.csv')
    flow.to_csv (dir + 'sensors4/flow.csv')

def concat_all():
    """合成最终的速度和流量矩阵"""
    speed_all = pd.DataFrame ()
    flow_all = pd.DataFrame ()
    for i in range (1, 5):
        suffix_speed = 'sensors%s/speed.csv' % i
        suffix_flow = 'sensors%s/flow.csv' % i
        speed = pd.read_csv (dir + suffix_speed, index_col=0)
        flow = pd.read_csv (dir + suffix_flow, index_col=0)
        speed_all = pd.concat ([speed_all, speed], axis=1)
        flow_all = pd.concat ([flow_all, flow], axis=1)
        print (speed_all.shape)
        print (flow_all.shape)
    speed_all.to_csv (dir + 'all_speed.csv')
    flow_all.to_csv (dir + 'all_flow.csv')


# concat_slice()
# Split()
# fill_sensors4()
# concat_all()
#
speed = pd.read_csv(dir + 'HuaNan_all_speed.csv',index_col= 0)
adj = np.zeros([speed.shape[1],speed.shape[1]],dtype= int)
lanes = 4
print(adj.shape[0])
for i in range(0,adj.shape[0]):
    adj[i][i] = 1
    if i + 4 < adj.shape[0] :
        adj[i][i+4] = 1
    if i - 4 >= 0 :
        adj[i][i-4] = 1
    if i % lanes - 1 >= 0 :
        adj[i][i - 1] = 1
    if i % lanes + 1 < lanes :
        adj[i][i + 1] = 1
adj = pd.DataFrame(adj)
adj.to_csv('/root/Lane_level/AGWN/Data_process/data/Huanan_adj.csv')
print(speed)
