import sqlite3 as sql

from config.config import net_db_path, identifiers

iden_default = 'KR_ComputerScience'

query = """
    SELECT name
    FROM Edges
    WHERE field = ? AND column5 > ?
"""


def generate_query(iden, record_type='kr', gender='m'):

    assert (record_type in ['kr', 'us', 'overall'])
    assert (gender in ['f', 'm', 'overall'])

    field = iden.split('_')[1]

    if field == 'ComputerScience':
        field = 'Computer\nScience'

    query_base = f"""
        SELECT name
        FROM Edges
        WHERE field = {field} AND column5 > ?
    """

    match record_type:
        case 'kr':
            query_type = ' AND aarc_extension != 1
        case 'us':
            query_type = ' AND aarc_extension >0'
        case 'overall':
            query_type = ''
        case _:
            assert (False)

    query_base += query_type

     match gender:
        case 'kr':
            query_gender = ' AND gender '
        case 'us':
            query_gender = ' AND aarc_extension >0'
        case 'overall':
            query_type = ''
        case _:
            assert (False)

    return query


db_path = net_db_path("KR_Integrated")
conn = sql.connect(db_path)
cursor = conn.cursor()