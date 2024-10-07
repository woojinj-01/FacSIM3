import sqlite3 as sql

from config.config import net_db_path
import parse.queries as qs


def grant_nations_to_edge(g):

    iden = g.name

    db_path = net_db_path(iden)

    conn = sql.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(Edges);")
    columns_info = [col[1] for col in cursor.fetchall()]

    add_src_nation = '''
        ALTER TABLE Edges
        ADD COLUMN src_nation TEXT
        '''
    
    add_dst_nation = '''
        ALTER TABLE Edges
        ADD COLUMN dst_nation TEXT
        '''
    
    if 'src_nation' not in columns_info:
        cursor.execute(add_src_nation)

    if 'dst_nation' not in columns_info:
        cursor.execute(add_dst_nation)

    grant_src_nation = '''
        UPDATE Edges
        SET src_nation = ?
        WHERE id = ?
        '''
    
    grant_dst_nation = '''
        UPDATE Edges
        SET dst_nation = ?
        WHERE id = ?
        '''

    for src_id, dst_id, data in g.edges(data=True):

        edge_id = data["id"]
        src_nation = g.nodes[src_id]['nation']
        dst_nation = g.nodes[dst_id]['nation']

        cursor.execute(grant_src_nation, (src_nation, edge_id))
        cursor.execute(grant_dst_nation, (dst_nation, edge_id))

    conn.commit()
    cursor.close()
    conn.close()


if __name__ == "__main__":

    from networks.construct_net import construct_network

    nets = construct_network()
    
    for g in nets.values():
        grant_nations_to_edge(g)


