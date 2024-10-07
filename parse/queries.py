SELECT_TABLE_NODES = '''
    SELECT * FROM Nodes 
    '''

SELECT_TABLE_EDGES = '''
    SELECT * FROM Edges 
    '''

CREATE_TABLE_NODES = '''
    CREATE TABLE IF NOT EXISTS Nodes (
        id INTEGER PRIMARY KEY,
        name TEXT,
        nation TEXT,
        num_fac_in INTEGER,
        num_fac_out INTEGER
        )
    '''

CREATE_TABLE_EDGES = '''
        CREATE TABLE IF NOT EXISTS Edges (
            id INTEGER PRIMARY KEY,
            src TEXT,
            dst TEXT,
            src_id INTEGER,
            dst_id INTEGER,
            gender TEXT,
            phd_start_year INTEGER,
            phd_end_year INTEGER,
            ap_start_year INTEGER,
            ap_end_year INTEGER,
            FOREIGN KEY (src_id) REFERENCES Nodes(id),
            FOREIGN KEY (dst_id) REFERENCES Nodes(id)
        )
    '''

INSERT_NODE = '''
    INSERT INTO Nodes (name, nation, num_fac_in, num_fac_out)
    VALUES (?, ?, ?, ?)
    '''

INSERT_EDGE = '''
    INSERT INTO Edges (src, dst, src_id, dst_id, gender, phd_start_year, phd_end_year, ap_start_year, ap_end_year)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''

INCRE_IN_EDGE = '''
    UPDATE Nodes
    SET num_fac_in = num_fac_in + 1
    WHERE name = ?
'''

INCRE_OUT_EDGE = '''
    UPDATE Nodes
    SET num_fac_out = num_fac_out + 1
    WHERE name = ?
'''

GET_MAX_DOMESTICRANK = '''SELECT MAX(rank_domestic) AS MaxValue FROM Nodes'''

GET_NODEID_BY_NAME = '''
    SELECT id
    FROM Nodes
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

GET_DOMESTICRANK_BY_NODEID = '''
    SELECT rank_domestic
    FROM Nodes
    WHERE id = ?
    '''

GET_GLOBALRANK_BY_NODEID = '''
    SELECT rank_global
    FROM Nodes
    WHERE id = ?
    '''

GET_WAPMANRANK_BY_NODEID = '''
    SELECT rank_wapman
    FROM Nodes
    WHERE id = ?
    '''

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

GRANT_GLOBAL_RANK_BY_NODEID = '''
    UPDATE Nodes
    SET rank_global = ?
    WHERE id = ?
    '''

GRANT_DOMESTIC_RANK_BY_NODEID = '''
    UPDATE Nodes
    SET rank_domestic = ?
    WHERE id = ?
    '''

GRANT_WAPMANRANK_BY_NODEID = '''
    UPDATE Nodes
    SET rank_wapman = ?
    WHERE id = ?
    '''

GRANT_WAPMANRANK_BY_NAME = '''
    UPDATE Nodes
    SET rank_wapman = ?
    WHERE name = ?
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
