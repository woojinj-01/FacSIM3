def get_abbrev(iden):

    match iden:
        case 'KR_Biology':
            return 'bi'
        case 'KR_ComputerScience':
            return 'cs'
        case 'KR_Physics':
            return 'ph'
        case _:
            return None
        

SELECT_TABLE_NODES = '''
    SELECT * FROM Nodes 
    '''

SELECT_TABLE_EDGES = '''
    SELECT * FROM Edges 
    '''

SELECT_EDGES_FOR_GNET = '''
    SELECT * FROM Edges 
    WHERE field = ? AND phd_inst_id IS NOT NULL AND which_job_is_ap IS NOT NULL;
    '''

CREATE_TABLE_NODES = '''
    CREATE TABLE IF NOT EXISTS Nodes (
        id INTEGER PRIMARY KEY,
        name TEXT,
        nation TEXT,
        bi_rank_global INTEGER,
        bi_rank_domestic INTEGER,
        bi_rank_wapman INTEGER,
        cs_rank_global INTEGER,
        cs_rank_domestic INTEGER,
        cs_rank_wapman INTEGER,
        ph_rank_global INTEGER,
        ph_rank_domestic INTEGER,
        ph_rank_wapman INTEGER,
        address TEXT,
        latitude FLOAT,
        longitude FLOAT
        )
    '''

CREATE_TABLE_EDGES = '''
        CREATE TABLE IF NOT EXISTS Edges (
            id INTEGER PRIMARY KEY,
            name TEXT,
            field TEXT,
            institution TEXT,
            department TEXT,
            current_position TEXT,
            gender TEXT,
            bs_inst_name TEXT,
            bs_inst_id INTEGER,
            bs_nation TEXT,
            bs_start_year INTEGER,
            bs_end_year INTEGER,
            ms_inst_name TEXT,
            ms_inst_id INTEGER,
            ms_nation TEXT,
            ms_start_year INTEGER,
            ms_end_year INTEGER,
            phd_inst_name TEXT,
            phd_inst_id INTEGER,
            phd_nation TEXT,
            phd_start_year INTEGER,
            phd_end_year INTEGER,
            job1_type TEXT,
            job1_inst_name TEXT,
            job1_inst_id INTEGER,
            job1_nation TEXT,
            job1_start_year INTEGER,
            job1_end_year INTEGER,
            job2_type TEXT,
            job2_inst_name TEXT,
            job2_inst_id INTEGER,
            job2_nation TEXT,
            job2_start_year INTEGER,
            job2_end_year INTEGER,
            job3_type TEXT,
            job3_inst_name TEXT,
            job3_inst_id INTEGER,
            job3_nation TEXT,
            job3_start_year INTEGER,
            job3_end_year INTEGER,
            job4_type TEXT,
            job4_inst_name TEXT,
            job4_inst_id INTEGER,
            job4_nation TEXT,
            job4_start_year INTEGER,
            job4_end_year INTEGER,
            job5_type TEXT,
            job5_inst_name TEXT,
            job5_inst_id INTEGER,
            job5_nation TEXT,
            job5_start_year INTEGER,
            job5_end_year INTEGER,
            job6_type TEXT,
            job6_inst_name TEXT,
            job6_inst_id INTEGER,
            job6_nation TEXT,
            job6_start_year INTEGER,
            job6_end_year INTEGER,
            job7_type TEXT,
            job7_inst_name TEXT,
            job7_inst_id INTEGER,
            job7_nation TEXT,
            job7_start_year INTEGER,
            job7_end_year INTEGER,
            job8_type TEXT,
            job8_inst_name TEXT,
            job8_inst_id INTEGER,
            job8_nation TEXT,
            job8_start_year INTEGER,
            job8_end_year INTEGER,
            job9_type TEXT,
            job9_inst_name TEXT,
            job9_inst_id INTEGER,
            job9_nation TEXT,
            job9_start_year INTEGER,
            job9_end_year INTEGER,
            job10_type TEXT,
            job10_inst_name TEXT,
            job10_inst_id INTEGER,
            job10_nation TEXT,
            job10_start_year INTEGER,
            job10_end_year INTEGER,
            which_job_is_ap INTEGER,
            FOREIGN KEY (bs_inst_id) REFERENCES Nodes(id),
            FOREIGN KEY (ms_inst_id) REFERENCES Nodes(id),
            FOREIGN KEY (phd_inst_id) REFERENCES Nodes(id),
            FOREIGN KEY (job1_inst_id) REFERENCES Nodes(id),
            FOREIGN KEY (job2_inst_id) REFERENCES Nodes(id),
            FOREIGN KEY (job3_inst_id) REFERENCES Nodes(id),
            FOREIGN KEY (job4_inst_id) REFERENCES Nodes(id),
            FOREIGN KEY (job5_inst_id) REFERENCES Nodes(id),
            FOREIGN KEY (job6_inst_id) REFERENCES Nodes(id),
            FOREIGN KEY (job7_inst_id) REFERENCES Nodes(id),
            FOREIGN KEY (job8_inst_id) REFERENCES Nodes(id),
            FOREIGN KEY (job9_inst_id) REFERENCES Nodes(id),
            FOREIGN KEY (job10_inst_id) REFERENCES Nodes(id)
        )
    '''

INSERT_NODE = '''
    INSERT INTO Nodes (name, nation)
    VALUES (?, ?)
    '''

INSERT_EDGE = '''
    INSERT INTO Edges (name,
    field,
    institution,
    department,
    current_position,
    gender)
    VALUES (?, ?, ?, ?, ?, ?)
    '''

INSERT_NODE_AARC = '''
    INSERT INTO Nodes (name, nation, aarc_extension)
    VALUES (?, ?, ?)
    '''

INSERT_EDGE_AARC = '''
    INSERT INTO Edges (name,
    field,
    gender,
    phd_inst_name,
    phd_nation,
    phd_inst_id,
    phd_end_year,
    job1_type,
    job1_inst_name,
    job1_nation,
    job1_inst_id,
    which_job_is_ap,
    aarc_extension)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''

def SET_FIELD_BY_EDGENAME(field):

    query = f'''
    UPDATE Edges
    SET {field} = ?
    WHERE name =?
    '''

    return query


def INCRE_FIELD_BY_NODEID(field):

    query = f'''
    UPDATE Nodes
    SET {field} = {field} + 1
    WHERE id =?
    '''

    return query


def GET_MAX_DRANK(iden):
    
    abbrev = get_abbrev(iden)

    query = f'''
    SELECT MAX({abbrev}_rank_domestic) AS MaxValue FROM Nodes
    '''

    return query

GET_NODEID_BY_NAME = '''
    SELECT id
    FROM Nodes
    WHERE name = ?
    '''

GET_EDGEID_BY_NAME = '''
    SELECT id
    FROM Edges
    WHERE name = ?
    '''

GET_NAME_BY_NODEID = '''
    SELECT name
    FROM Nodes
    WHERE id = ?
    '''

GET_NATION_BY_NODEID = '''
    SELECT nation
    FROM Nodes
    WHERE id = ?
    '''


def GET_DRANK_BY_NODEID(iden):

    abbrev = get_abbrev(iden)

    query = f'''
    SELECT {abbrev}_rank_domestic
    FROM Nodes
    WHERE id = ?
    '''

    return query


def GET_GRANK_BY_NODEID(iden):

    abbrev = get_abbrev(iden)

    query = f'''
    SELECT {abbrev}_rank_global
    FROM Nodes
    WHERE id = ?
    '''

    return query


def GET_WRANK_BY_NODEID(iden):

    abbrev = get_abbrev(iden)

    query = f'''
    SELECT {abbrev}_rank_wapman
    FROM Nodes
    WHERE id = ?
    '''

    return query

GET_ADDRESS_BY_NODEID = '''
    SELECT address
    FROM Nodes
    WHERE id = ?
    '''

GET_LATITUDE_BY_NODEID = '''
    SELECT latitude
    FROM Nodes
    WHERE id = ?
    '''

GET_LONGITUDE_BY_NODEID = '''
    SELECT longitude
    FROM Nodes
    WHERE id = ?
    '''

GET_SRCID_BY_EDGEID = '''
    SELECT src_id
    FROM Edges
    WHERE id = ?
    '''

GET_DSTID_BY_EDGEID = '''
    SELECT dst_id
    FROM Edges
    WHERE id = ?
    '''

GET_KR2KR_EDGES = '''
    SELECT *
    FROM Edges
    WHERE src_nation = 'KR' and dst_nation = 'KR'
    '''

GET_US2KR_EDGES = '''
    SELECT *
    FROM Edges
    WHERE src_nation = 'US' and dst_nation = 'KR'
    '''

GET_KR2US_EDGES = '''
    SELECT *
    FROM Edges
    WHERE src_nation = 'KR' and dst_nation = 'US'
    '''


def GRANT_GRANK_BY_NODEID(iden):

    abbrev =get_abbrev(iden)

    query = f'''
        UPDATE Nodes
        SET {abbrev}_rank_global = ?
        WHERE id = ?
    '''

    return query


def GRANT_DRANK_BY_NODEID(iden):

    abbrev =get_abbrev(iden)

    query = f'''
        UPDATE Nodes
        SET {abbrev}_rank_domestic = ?
        WHERE id = ?
    '''

    return query


def GRANT_WRANK_BY_NODEID(iden):

    abbrev =get_abbrev(iden)

    query = f'''
        UPDATE Nodes
        SET {abbrev}_rank_wapman = ?
        WHERE id = ?
    '''

    return query


ADD_AARC_COL_NODES = '''
    ALTER TABLE Nodes
    ADD COLUMN aarc_extension INTEGER
    '''

ADD_AARC_COL_EDGES = '''
    ALTER TABLE Edges
    ADD COLUMN aarc_extension INTEGER
    '''

ADD_ADDRESS_COL = '''
    ALTER TABLE Nodes
    ADD COLUMN address TEXT
    '''

ADD_LATITUDE_COL = '''
    ALTER TABLE Nodes
    ADD COLUMN latitude float
    '''

ADD_LONGITUDE_COL = '''
    ALTER TABLE Nodes
    ADD COLUMN longitude float
    '''

SET_ADDRESS_BY_NODEID = '''
    UPDATE Nodes
    SET address = ?
    WHERE id =?
    '''

SET_LATITUDE_BY_NODEID = '''
    UPDATE Nodes
    SET latitude = ?
    WHERE id =?
    '''

SET_LONGITUDE_BY_NODEID = '''
    UPDATE Nodes
    SET longitude = ?
    WHERE id =?
    '''

SET_DISTANCE_BY_NODEID = '''
    UPDATE Nodes
    SET distance_to_seoul = ?
    WHERE id =?
    '''

SET_ADDRESS_BY_NAME = '''
    UPDATE Nodes
    SET address = ?
    WHERE name =?
    '''

SET_LATITUDE_BY_NAME = '''
    UPDATE Nodes
    SET latitude = ?
    WHERE name =?
    '''

SET_LONGITUDE_BY_NAME = '''
    UPDATE Nodes
    SET longitude = ?
    WHERE name =?
    '''
