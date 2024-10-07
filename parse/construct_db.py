import sqlite3 as sql
import pandas as pd

import parse.queries as qs
from parse.text_processing import normalize_inst_name

from config.config import xlsx_data_path, raw_db_path, net_db_path, identifiers


class RawDBParser:
    def __init__(self, arg_raw_db_path, arg_net_db_path) -> None:

        self.conn_raw = sql.connect(arg_raw_db_path)
        self.cursor_raw = self.conn_raw.cursor()

        self.conn_net = sql.connect(arg_net_db_path)
        self.cursor_net = self.conn_net.cursor()

        self.cursor_net.execute(qs.CREATE_TABLE_NODES)
        self.cursor_net.execute(qs.CREATE_TABLE_EDGES)

        self.conn_net.execute('PRAGMA foreign_keys = ON')
        
        self.insts = set()
        self.num_faculty = 0

    def parse(self):
        
        self.cursor_raw.execute("SELECT name FROM sqlite_master WHERE type='table';")

        table_names = [row[0] for row in self.cursor_raw.fetchall()]
        
        for name in table_names:
            self.parse_table(name)

        self.conn_raw.commit()
        self.conn_raw.close()
        self.conn_net.commit()
        self.conn_net.close()

    def parse_table(self, table_name):

        print(f"Parsing {table_name}")

        self.cursor_raw.execute(f"PRAGMA table_info({table_name});")

        self.cursor_raw.execute(f"SELECT * FROM {table_name}")
        rows = self.cursor_raw.fetchall()

        self.cursor_raw.execute(f"PRAGMA table_info({table_name});")

        for row in rows:

            phd_inst, phd_start, phd_end = self.find_phd_inst(row)
            ap_inst, ap_start, ap_end = self.find_ap_inst(row)

            if phd_inst is None or ap_inst is None:
                continue  # invalid record
            
            # valid record

            self.num_faculty += 1
            id = self.num_faculty

            gender = self.find_gender(row, table_name)

            # For debugging purpose
            # print(f"=== Id: {id} ===")
            # print(f"Gender: {gender}")
            # print(f"Ph.D. at {phd_inst} from {phd_start} to {phd_end}")
            # print(f"AP at {ap_inst} from {ap_start} to {ap_end}")

            # add node

            if phd_inst not in self.insts:
                self.insts.add(phd_inst)
                self.cursor_net.execute(qs.INSERT_NODE,
                                        (phd_inst,
                                         phd_inst.split(',')[2].strip().upper(), 0, 0))
                
            if ap_inst not in self.insts:
                self.insts.add(ap_inst)
                self.cursor_net.execute(qs.INSERT_NODE,
                                        (ap_inst,
                                         ap_inst.split(',')[2].strip().upper(), 0, 0))

            self.cursor_net.execute(qs.INCRE_OUT_EDGE, (phd_inst, ))
            self.cursor_net.execute(qs.INCRE_IN_EDGE, (ap_inst, ))

            self.cursor_net.execute(qs.GET_NODEID_BY_NAME, (phd_inst, ))
            phd_inst_id = self.cursor_net.fetchone()[0]

            self.cursor_net.execute(qs.GET_NODEID_BY_NAME, (ap_inst, ))
            ap_inst_id = self.cursor_net.fetchone()[0]

            # add edge
            self.cursor_net.execute(qs.INSERT_EDGE, (phd_inst, ap_inst, phd_inst_id, ap_inst_id, gender, phd_start, phd_end, ap_start, ap_end))

    def find_phd_inst(self, row):

        for i in range(len(row)):

            if not isinstance(row[i], str):
                continue

            if row[i].lower() in ['phd', 'ph.d']:

                inst = row[i + 2]
                start = row[i - 2]
                end = row[i - 1]

                inst = normalize_inst_name(inst)

                if inst is not None:

                    start_year = start if isinstance(start, int)\
                        else int(start) if isinstance(start, float)\
                        else int(start) if start.isdigit() else 0
                    
                    end_year = end if isinstance(end, int)\
                        else int(end) if isinstance(end, float)\
                        else int(end) if end.isdigit() else 0

                    return inst, start_year, end_year
                else:
                    continue

        return None, None, None

    def find_ap_inst(self, row):

        for i in range(len(row)):

            if not isinstance(row[i], str):
                continue

            if row[i].lower() == 'assistant professor':

                inst = row[i + 1]
                start = row[i - 2]
                end = row[i - 1]

                inst = normalize_inst_name(inst)

                if inst is not None:

                    start_year = start if isinstance(start, int)\
                        else int(start) if isinstance(start, float)\
                        else int(start) if start.isdigit() else 0
                    
                    end_year = end if isinstance(end, int)\
                        else int(end) if isinstance(end, float)\
                        else int(end) if end.isdigit() else 0

                    return inst, start_year, end_year
                else:
                    continue

        return None, None, None
    
    def find_gender(self, row, table_name):

        self.cursor_raw.execute(f"PRAGMA table_info({table_name});")
        columns = self.cursor_raw.fetchall()

        gender_index = -1

        for i, column in enumerate(columns):
            if isinstance(column[1], str) and column[1].lower() == 'sex':
                gender_index = i
                break

        if gender_index == -1:
            return "Not found"
        
        gender = row[gender_index]

        if not isinstance(gender, str):
            print(gender)
            return "Not found"
        
        elif gender.lower() in ["female", "f"]:
            return "F"

        elif gender.lower() in ["male", "m"]:
            return "M"

        else:
            print(gender)
            return "Not found"


def _xlsx_to_db(iden):

    xlsx_path = xlsx_data_path(iden)
    conn = sql.connect(raw_db_path(iden))

    dfs = pd.read_excel(xlsx_path, sheet_name=None)

    for name, df in dfs.items():

        name.replace(' ', '_')
        df.to_sql(name, conn, if_exists='append', index=False)

    conn.commit()
    conn.close


def construct_raw_db():

    for iden in identifiers:
        _xlsx_to_db(iden)


def construct_net_db(iden):

    raw_db_name = raw_db_path(iden)
    net_db_name = net_db_path(iden)

    print(raw_db_name)

    parser = RawDBParser(raw_db_name, net_db_name)

    parser.parse()


if __name__ == "__main__":

    for iden in identifiers:
        construct_net_db(iden)