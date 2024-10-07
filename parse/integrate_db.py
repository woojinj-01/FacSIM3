import sqlite3 as sql
import pandas as pd

import parse.queries_integration as qsi
from parse.text_processing import normalize_inst_name, normalize_job_name, normalize_fac_name, normalize_general

from config.config import xlsx_data_path, raw_db_path, net_db_path, identifiers, csv_data_path
        

class RawDBIntegrator():
    def __init__(self, arg_net_db_path) -> None:

        self.conn_cursor = []

        for iden in identifiers:
            raw_db_name = raw_db_path(iden)

            conn_raw = sql.connect(raw_db_name)
            cursor_raw = conn_raw.cursor()

            self.conn_cursor.append((iden, conn_raw, cursor_raw))

        self.conn_net = sql.connect(arg_net_db_path)
        self.cursor_net = self.conn_net.cursor()

        self.cursor_net.execute(qsi.CREATE_TABLE_NODES)
        self.cursor_net.execute(qsi.CREATE_TABLE_EDGES)

        self.conn_net.execute('PRAGMA foreign_keys = ON')
        
        self.insts = set()
        self.insts.add(None)
        
        self.num_records = 0
        self.current_iden = None

    def _get_abbrev_by_iden(self, iden):

        match iden:
            case 'KR_Biology':
                return 'bi'
            case 'KR_ComputerScience':
                return 'cs'
            case 'KR_Physics':
                return 'ph'
            case _:
                return None
            
    def _get_field_by_iden(self, iden):

        match iden:
            case 'KR_Biology':
                return 'Biology'
            case 'KR_ComputerScience':
                return 'Computer Science'
            case 'KR_Physics':
                return 'Physics'
            case _:
                return None
            
    def parse(self):

        GET_TABLE_NAMES = "SELECT name FROM sqlite_master WHERE type='table';"
        
        for iden, conn_raw, cursor_raw in self.conn_cursor:

            self.current_iden = iden
        
            cursor_raw.execute(GET_TABLE_NAMES)

            table_names = [row[0] for row in cursor_raw.fetchall()]
            
            for name in table_names:
                self.parse_table(cursor_raw, name)

            conn_raw.commit()
            cursor_raw.close()
            conn_raw.close()

        self.current_iden = None

        self.conn_net.commit()
        self.cursor_net.close()
        self.conn_net.close()

    def parse_table(self, cursor_raw, table_name):

        print(f"Parsing {table_name}")

        cursor_raw.execute(f"PRAGMA table_info({table_name});")
        column_names = [col_info[1] for col_info in cursor_raw.fetchall()]

        cursor_raw.execute(f"SELECT * FROM {table_name}")
        rows = cursor_raw.fetchall()

        for row in rows:
            
            bs_inst, bs_nation, bs_start, bs_end = self.find_bs_inst(row)
            ms_inst, ms_nation, ms_start, ms_end = self.find_ms_inst(row)
            phd_inst, phd_nation, phd_start, phd_end = self.find_phd_inst(row)

            jobs = self.find_job_inst(row, column_names)

            gender = self.find_gender(row, table_name, column_names)

            fac_name, current_inst, current_dep, current_pos = self.find_others(row, column_names)

            fac_name = normalize_fac_name(fac_name)
            current_inst = normalize_inst_name(current_inst)
            current_dep = normalize_general(current_dep)
            current_pos = normalize_job_name(current_pos)

            self.num_records += 1

            # For debugging purpose
            # print(f"=== Id: {id} ===")
            # print(f"Gender: {gender}")
            # print(f"Ph.D. at {phd_inst} from {phd_start} to {phd_end}")
            # print(f"AP at {ap_inst} from {ap_start} to {ap_end}")

            # add node

            if bs_inst not in self.insts:
                self.insts.add(bs_inst)
                self.cursor_net.execute(qsi.INSERT_NODE,
                                        (bs_inst, bs_nation))
                
            if ms_inst not in self.insts:
                self.insts.add(ms_inst)
                self.cursor_net.execute(qsi.INSERT_NODE, (ms_inst, ms_nation))

            if phd_inst not in self.insts:
                self.insts.add(phd_inst)
                self.cursor_net.execute(qsi.INSERT_NODE, (phd_inst, phd_nation))
                
            for job_info in jobs:

                _, inst, nation, _, _ = job_info

                if inst not in self.insts:
                    self.insts.add(inst)
                    self.cursor_net.execute(qsi.INSERT_NODE, (inst, nation))

            self.conn_net.commit()

            self.cursor_net.execute(qsi.GET_NODEID_BY_NAME, (bs_inst, ))
            info = self.cursor_net.fetchone()

            bs_inst_id = info[0] if info is not None else None

            self.cursor_net.execute(qsi.GET_NODEID_BY_NAME, (ms_inst, ))
            info = self.cursor_net.fetchone()

            ms_inst_id = info[0] if info is not None else None

            self.cursor_net.execute(qsi.GET_NODEID_BY_NAME, (phd_inst, ))
            info = self.cursor_net.fetchone()

            phd_inst_id = info[0] if info is not None else None

            self.cursor_net.execute(qsi.INSERT_EDGE, (fac_name, self._get_field_by_iden(self.current_iden),
                                                      current_inst, current_dep, current_pos, gender))
            
            if bs_inst is not None:
                self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME("bs_inst_name"), (bs_inst, fac_name))
                self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME("bs_inst_id"), (bs_inst_id, fac_name))
                self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME("bs_nation"), (bs_nation, fac_name))
                self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME("bs_start_year"), (bs_start, fac_name))
                self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME("bs_end_year"), (bs_end, fac_name))

            if ms_inst is not None:
                self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME("ms_inst_name"), (ms_inst, fac_name))
                self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME("ms_inst_id"), (ms_inst_id, fac_name))
                self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME("ms_nation"), (ms_nation, fac_name))
                self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME("ms_start_year"), (ms_start, fac_name))
                self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME("ms_end_year"), (ms_end, fac_name))

            if phd_inst is not None:
                self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME("phd_inst_name"), (phd_inst, fac_name))
                self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME("phd_inst_id"), (phd_inst_id, fac_name))
                self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME("phd_nation"), (phd_nation, fac_name))
                self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME("phd_start_year"), (phd_start, fac_name))
                self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME("phd_end_year"), (phd_end, fac_name))

            job_index = 1

            ap_pos = None

            for job_info in jobs:

                job, inst, nation, start, end = job_info

                field = f"job{job_index}"

                if job == 'Assistant professor':
                    ap_pos = job_index

                self.cursor_net.execute(qsi.GET_NODEID_BY_NAME, (inst, ))
                inst_id = self.cursor_net.fetchone()[0]

                if inst is not None:
                    self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME(f"{field}_type"), (job, fac_name))
                    self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME(f"{field}_inst_name"), (inst, fac_name))
                    self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME(f"{field}_inst_id"), (inst_id, fac_name))
                    self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME(f"{field}_nation"), (nation, fac_name))
                    self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME(f"{field}_start_year"), (start, fac_name))
                    self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME(f"{field}_end_year"), (end, fac_name))

                job_index += 1

            if ap_pos is not None:
                self.cursor_net.execute(qsi.SET_FIELD_BY_EDGENAME("which_job_is_ap"), (ap_pos, fac_name))
                

    def find_bs_inst(self, row):

        target = ['bs']

        return self._find_degree_inst(row, target)

    def find_ms_inst(self, row):

        target = ['ms']

        return self._find_degree_inst(row, target)

    def find_phd_inst(self, row):

        target = ['phd', 'ph.d']

        return self._find_degree_inst(row, target)

    def _find_degree_inst(self, row, target):

        for i in range(len(row)):

            if not isinstance(row[i], str):
                continue

            if row[i].lower() in target:

                inst = row[i + 2]
                start = row[i + 3]
                end = row[i + 4]

                inst = normalize_inst_name(inst)

                if inst is not None:

                    start_year = start if isinstance(start, int)\
                        else int(start) if isinstance(start, float)\
                        else int(start) if start.isdigit() else None
                    
                    end_year = end if isinstance(end, int)\
                        else int(end) if isinstance(end, float)\
                        else int(end) if end.isdigit() else None
                    
                    nation = inst.split(', ')[2]

                    return inst, nation, start_year, end_year
                else:
                    continue

        return None, None, None, None

    def find_job_inst(self, row, column_names):

        jobs = []

        for i in range(10):

            col_name = f"Job.{i}" if i != 0 else "Job"

            if col_name in column_names:
                index = column_names.index(col_name)
                found = self._find_job_inst(row, index)

                if found is not None:
                    jobs.append(found)
                
            else:
                break
        
        return jobs

    def _find_job_inst(self, row, i):

        if not isinstance(row[i], str):
            return None

        job = normalize_job_name(row[i])
        inst = row[i + 1]
        start = row[i + 2]
        end = row[i + 3]

        inst = normalize_inst_name(inst)

        if inst is not None:

            start_year = 0 if start is None\
                else start if isinstance(start, int)\
                else int(start) if isinstance(start, float)\
                else int(start) if start.isdigit() else 0
            
            end_year = 0 if end is None\
                else end if isinstance(end, int)\
                else int(end) if isinstance(end, float)\
                else int(end) if end.isdigit() else 0
            
            nation = inst.split(', ')[2]

            return job, inst, nation, start_year, end_year
        else:
            return None

    def find_gender(self, row, table_name, columns):

        for col in ['sex', 'Sex', 'gender', 'Gender']:
            if col in columns:
                index = columns.index(col)
                gender = row[index]

                if not isinstance(gender, str):
                    return None
                
                elif gender.lower() in ["female", "f"]:
                    return "Female"

                elif gender.lower() in ["male", "m"]:
                    return "Male"

                else:
                    return None
            
        return None

    def find_others(self, row, column_names):

        found = []

        for field in ['Faculty name', 'Institution', 'Department', 'Current Rank']:

            if field in column_names:
                index = column_names.index(field)

                if isinstance(row[index], str):
                    found.append(row[index])
                else:
                    found.append(None)

            else:
                found.append(None)

        return tuple(found)


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


def integrate_db():

    from datetime import datetime

    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")

    net_db_name = net_db_path(f"KR_Integrated_{current_time}")

    parser = RawDBIntegrator(net_db_name)

    parser.parse()


def extend_db_aarc():

    import math

    conn = sql.connect(net_db_path('KR_Integrated'))
    cursor = conn.cursor()

    cursor.execute(qsi.ADD_AARC_COL_NODES)
    cursor.execute(qsi.ADD_AARC_COL_EDGES)

    df = pd.read_csv(csv_data_path('AARC_Extension'))
    
    query = "SELECT name FROM Nodes;"
    cursor.execute(query)
    results = cursor.fetchall()
    
    # Extract the column values from the results
    insts = set([row[0] for row in results])
    insts.add(None)

    for row in df.iterrows():
        
        field = row[1]['Field'] 
        name = row[1]['PersonName']
        gender = row[1]['Gender']

        if gender == 'Unknown':
            gender = None

        phd_inst = normalize_inst_name(row[1]['DegreeInstitutionName'])
        phd_end_year = row[1]['DegreeYear']

        if math.isnan(phd_end_year):
            phd_end_year = None
        else:
            phd_end_year = int(phd_end_year)

        ap_inst = normalize_inst_name(row[1]['InstitutionName'])

        if phd_inst not in insts:
            insts.add(phd_inst)
            print(phd_inst)
            cursor.execute(qsi.INSERT_NODE_AARC, (phd_inst, phd_inst.split(', ')[2], 1))

        if ap_inst not in insts:
            insts.add(ap_inst)
            print(ap_inst)
            cursor.execute(qsi.INSERT_NODE_AARC, (ap_inst, ap_inst.split(', ')[2], 1))

        conn.commit()

        cursor.execute(qsi.GET_NODEID_BY_NAME, (phd_inst, ))
        phd_inst_id = cursor.fetchone()
        
        if phd_inst_id is not None:
            phd_inst_id = phd_inst_id[0]

        cursor.execute(qsi.GET_NODEID_BY_NAME, (ap_inst, ))
        ap_inst_id = cursor.fetchone()

        if ap_inst_id is not None:
            ap_inst_id = ap_inst_id[0]

        if phd_inst is not None:
            phd_nation = phd_inst.split(', ')[2]

        if ap_inst is not None:
            ap_nation = ap_inst.split(', ')[2]

        cursor.execute(qsi.INSERT_EDGE_AARC, (name, field, gender, phd_inst,
                                              phd_nation, phd_inst_id, phd_end_year,
                                              'Assistant professor', ap_inst, ap_nation, ap_inst_id, 1, 1))

    conn.commit()
    cursor.close()
    conn.close()


def modify_ap_pos():

    conn = sql.connect(net_db_path('KR_Integrated'))
    conn.row_factory = sql.Row
    cursor = conn.cursor()

    cursor.execute(f"PRAGMA table_info(Edges);")

    cursor.execute(f"SELECT * FROM Edges")
    rows = cursor.fetchall()

    results = {i: 0 for i in range(1, 11)}

    update_query = f"UPDATE Edges SET which_job_is_ap = ? WHERE id = ?"

    for row in rows:

        for i in range(1, 11):

            id = row["id"]
            job_type = row[f"job{i}_type"]

            if job_type is None:
                continue

            job_type_reg = job_type.lower().replace(" ", "") 

            if job_type_reg == 'assistantprofessor':
                results[i] += 1

                cursor.execute(update_query, (i, id))

                break

    conn.commit()
    cursor.close()
    conn.close()


if __name__ == "__main__":
    pass