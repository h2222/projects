# -*- coding: utf-8 -*-
from exceptions import ProcessIdNotFoundError
import pandas as pd
import numpy as np


class DM_Module(object):
    def __init__(self):
        # self.host = model_param['fileParam']['database']['host']
        # self.user = model_param['fileParam']['database']['user']
        # self.passwd = model_param['fileParam']['database']['password']
        # self.db_name = model_param['fileParam']['database']['database']
        # self.table_name = model_param['fileParam']['database']['proc_table']

        # pool = PooledDB(pymysql, 5, host=self.host, user=self.user, passwd=self.passwd,
        #                 db=self.db_name, port=3306, charset="utf8")  # 5为连接池里的最少连接数
        # conn = pool.connection()
        # sql = 'select * from {}'.format(self.table_name)
        # df = pd.read_sql(sql, con=conn)
        # conn.close()
        df = pd.read_csv("procs.csv")
        df["in_node"] = df["in_node"].astype(np.str)
        self.dm_dct = {}
        proc_group = df.groupby("processid")
        for pid, grp in proc_group:
            self.dm_dct[pid] = {}
            for in_node, gp in grp.groupby("in_node"):
                self.dm_dct[pid][in_node] = {}
                for i, row in gp.iterrows():
                    outnode = row.out_node
                    for st in row.semantic_type.split("+"):
                        self.dm_dct[pid][in_node][st] = outnode

    def predict(self, pid, in_node, intention):
        if pid in self.dm_dct:
            if (intention) and (intention in self.dm_dct[pid][in_node]):
                return self.dm_dct[pid][in_node][intention]
            else:
                return -96
        else:
            try:
                self.__init__()
                if (intention) and (intention in self.dm_dct[pid][in_node]):
                    return self.dm_dct[pid][in_node][intention]
                else:
                    return -96
            except KeyError:
                raise ProcessIdNotFoundError(pid)


if __name__ == "__main__":
    dm = DM_Module()
    print(dm.predict("benrenshoucui", "1.1", "肯定"))
