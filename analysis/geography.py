import googlemaps
import sqlite3 as sql
from haversine import haversine
import numpy as np

from config.config import net_db_path, identifiers, xlsx_data_path
import parse.queries_integration as qsi
from networks.construct_net import construct_network

GOOGLEMAPS_API_KEY = 'AIzaSyCQXy8R56hZHApnR318UpMJHyis-WDWyZ4'
seoul_coord = (37.532600, 127.024612)
busan_corod = (35.1379222, 129.05562775)

area_seoul = ['seoul']

area_capital = ['hwaseong', 'bucheon', 'suwon', 'ansan',
                'anseong', 'osan', 'yongin',
                'goyang', 'anyang', 'gyeonggi-do',
                'seongnam', '_x0008_suwon']

area_metro = ['songdo', 'incheon', 'daegu', 'gwangju', 'daejeon', 'ulsan', 'busan', 'pusan']

area_others = ['miryang', 'pohang', 'jeonju',
               'changwon', 'asan', 'gunsan', 'yeongju',
               'jeju', 'gongju', 'goesan',
               'jecheon', 'chuncheon', 'kyungsan',
               'gumi', 'masan', 'muan',
               'cheongju', 'yeosu', 'wonju',
               'cheonan', 'chuncheon-si', 'kunsan', 'sejong',
               'gyeongsan', 'jinju', 'suncheon', 'gangneung',
               'mokpo', 'cheonanl', 'sunchon', 'chungju',
               'gyeongju', 'kongju', 'andong',
               'gimhae', 'iksan', 'namwon']

areas = {'Seoul': area_seoul,
         'Capital area': area_capital,
         'Metropolitan cities': area_metro,
         'Others': area_others}

area_codes = {'seoul': 11, 'busan': 21, 'pusan': 21, 'daegu': 22,
            'incheon': 23,'gwangju': 24, 'kwangju': 24, 'daejeon': 25,
            'ulsan': 26, 'sejong': 29, 'gyeonggi-do': 31, 'gangwon-do': 32,
            'chungcheongbuk-do': 33, 'chungcheongnam-do': 34, 'jeollabuk-do': 35,
            'jeollanam-do': 36, 'gyeongsangbuk-do': 37, 'gyeongsangnam-do': 38,
            'jeju-do': 39}

max_domestic_ranks = {"Biology": 78,
                    "ComputerScience": 105,
                    "Physics": 65}


def classify_area(region):

    for label, area in areas.items():

        if region in area:
            return label
        
    return None


def get_column_values(cursor, table_name, column_name):
    # Query to select all data from the specified column
    query = f'SELECT {column_name} FROM {table_name}'
    cursor.execute(query)
    
    # Fetch all results
    results = cursor.fetchall()  # This returns a list of tuples

    # Convert list of tuples into a list of values
    values = [item[0] for item in results]
    return values


def site_to_address(site_name):

    maps = googlemaps.Client(key=GOOGLEMAPS_API_KEY)

    result = maps.geocode(site_name)

    if result:
        address = result[0]['formatted_address']
        lat = result[0]['geometry']['location']['lat']
        lng = result[0]['geometry']['location']['lng']

    else:
        address = None
        lat = None
        lng = None

    return address, lat, lng


def distance_between_sites(site1_coord, site2_coord):
    return haversine(site1_coord, site2_coord, unit='km')


def distance_to_seoul(site_coord):
    print(site_coord[0] - seoul_coord[0], site_coord[1] - seoul_coord[1])
    return distance_between_sites(site_coord, seoul_coord)


def grant_geoinfo_from_xlsx():

    import pandas as pd
    from parse.text_processing import normalize_inst_name

    xlsx_names = [xlsx_data_path(f'GeoInfo_Korea_{i + 1}') for i in range(4)]

    conn = sql.connect(net_db_path('KR_Integrated'))
    cursor = conn.cursor()

    for name in xlsx_names:

        df = pd.read_excel(name)

        for i, row in df.iterrows():
            
            row = list(row)
            
            inst_name = normalize_inst_name(row[0])
            address = row[1]
            address_modified = row[2]

            if isinstance(address_modified, str):
                address_to_grant = address_modified
            else:
                address_to_grant = address

            _, lat, lng = site_to_address(address_to_grant)

            cursor.execute(qsi.SET_ADDRESS_BY_NAME, (address_to_grant, inst_name))
            cursor.execute(qsi.SET_LATITUDE_BY_NAME, (lat, inst_name))
            cursor.execute(qsi.SET_LONGITUDE_BY_NAME, (lng, inst_name))

            # print(inst_name, address, address_modified, isinstance(address_modified, float))

            # cursor.execute(qsi.SET_ADDRESS_BY_NAME, (address_to_grant, inst_name))

    conn.commit()

    cursor.close()
    conn.close()


def grant_distance_to_seoul():

    conn = sql.connect(net_db_path('KR_Integrated'))
    cursor = conn.cursor()

    ids = get_column_values(cursor, 'Nodes', 'id')

    for id in ids:

        cursor.execute(qsi.GET_LATITUDE_BY_NODEID, (id,))
        lat = cursor.fetchone()[0]

        cursor.execute(qsi.GET_LONGITUDE_BY_NODEID, (id,))
        lng = cursor.fetchone()[0]

        if lat is None or lng is None:
            continue

        dist = distance_to_seoul((lat, lng))

        print(lat, lng, dist)
        cursor.execute(qsi.SET_DISTANCE_BY_NODEID, (dist, id))

    conn.commit()

    cursor.close()
    conn.close()


def grant_geoinfo(iden):

    conn = sql.connect(net_db_path(iden))
    cursor = conn.cursor()

    cursor.execute('PRAGMA table_info(Nodes)')

    cols_info = [col[1] for col in cursor.fetchall()]

    if 'address' not in cols_info:
        cursor.execute(qsi.ADD_ADDRESS_COL)
    
    if 'latitude' not in cols_info:
        cursor.execute(qsi.ADD_LATITUDE_COL)
    
    if 'longitude' not in cols_info:
        cursor.execute(qsi.ADD_LONGITUDE_COL)

    ids = get_column_values(cursor, 'Nodes', 'id')

    for id in ids:

        cursor.execute(qsi.GET_NAME_BY_NODEID, (id,))

        name = cursor.fetchall()

        address, lat, lng = site_to_address(name)

        cursor.execute(qsi.SET_ADDRESS_BY_NODEID, (address, id))
        cursor.execute(qsi.SET_LATITUDE_BY_NODEID, (lat, id))
        cursor.execute(qsi.SET_LONGITUDE_BY_NODEID, (lng, id))

    conn.commit()

    cursor.close()
    conn.close()


def grant_all_geoinfo():

    for iden in identifiers:
        grant_geoinfo(iden)


def radious_of_gyration(r_list, m_list):

    import numpy as np
    
    rs = np.array(r_list)
    ms = np.array(m_list)

    rs_squared = rs ** 2

    a = rs_squared @ ms
    b = np.sum(ms)

    if b == 0:
        return None
    else:
        return np.sqrt(a / b)
    

def calc_radg(dist_type='geo'):

    assert (dist_type in ['geo', 'rank'])

    nets = construct_network(net_type='domestic')
    results = {}

    for iden in identifiers:

        g = nets[iden] 
        results[iden] = _calc_radg(g, dist_type=dist_type)

    return results


def _calc_radg(g, dist_type='geo'):

    assert (dist_type in ['geo', 'rank'])

    iden = g.name
    abbrev = qsi.get_abbrev(iden)

    stats = {}

    for src_id, dst_id, data in g.edges(data=True):

        if src_id not in stats:
            stats[src_id] = {"r": [], "m": []}

        if dist_type == 'geo':
            src_lat = g.nodes[src_id]['latitude']
            src_lng = g.nodes[src_id]['longitude']

            dst_lat = g.nodes[dst_id]['latitude']
            dst_lng = g.nodes[dst_id]['longitude']

            if any(co is None for co in [src_lat, src_lng, dst_lat, dst_lng]):
                continue
            
            dist = distance_between_sites((src_lat, src_lng), (dst_lat, dst_lng))

        else:

            src_rank = g.nodes[src_id][f'{abbrev}_rank_domestic']
            dst_rank = g.nodes[dst_id][f'{abbrev}_rank_domestic']

            dist = src_rank - dst_rank

        stats[src_id]['r'].append(dist)
        stats[src_id]['m'].append(1)

    rads = {}

    for id, value in stats.items():

        rad = radious_of_gyration(value['r'], value['m'])

        if rad is not None:
            rads[id] = rad

    return rads


def calc_radg_ver2(dist_type='geo'):

    assert (dist_type in ['geo', 'rank'])

    nets = construct_network(net_type='domestic')
    results = {}

    for iden in identifiers:

        g = nets[iden] 
        results[iden] = _calc_radg_ver2(g, dist_type=dist_type)

    return results


def _calc_radg_ver2(g, dist_type='geo'):

    assert (dist_type in ['geo', 'rank'])

    iden = g.name
    abbrev = qsi.get_abbrev(iden)

    stats = {}

    for src_id, dst_id, data in g.edges(data=True):

        if src_id not in stats:
            stats[src_id] = []

        if dist_type == 'geo':
            dst_lat = g.nodes[dst_id]['latitude']
            dst_lng = g.nodes[dst_id]['longitude']

            if any(co is None for co in [dst_lat, dst_lng]):
                continue

            stats[src_id].append((dst_lat, dst_lng))

        else:
            dst_rank = g.nodes[dst_id][f'{abbrev}_rank_domestic']

            stats[src_id].append(dst_rank)
    
    cms = {}

    for id, value in stats.items():

        if len(value) == 0:
            continue

        if dist_type == 'geo':
            
            lats = [v[0] for v in value]
            lngs = [v[1] for v in value]

            cm = (np.mean(lats), np.mean(lngs))

        else:
            cm = np.mean(value)

        cms[id] = cm

    stats_cleaned = {}

    for id, value in stats.items():

        if id not in cms:
            continue

        cm = cms[id]

        if id not in stats_cleaned:
            stats_cleaned[id] = {"r": [], "m": []}

        for v in value:

            if dist_type == 'geo':
                dist = distance_between_sites(v, cm)

            else:
                dist = v - cm

            stats_cleaned[id]['r'].append(dist)
            stats_cleaned[id]['m'].append(1)

    rads = {}

    for id, value in stats_cleaned.items():

        rad = radious_of_gyration(value['r'], value['m'])

        if rad is not None:
            rads[id] = rad

    return rads


def calc_inst_by_region(net_type='global'):

    assert (net_type in ['global', 'domestic'])

    nets = construct_network(net_type=net_type)
    results = {}

    for iden in identifiers:

        g = nets[iden] 
        results[iden] = _calc_inst_by_region(g)

    return results


def _calc_inst_by_region(g):

    counts = {'Seoul': 0,
             'Capital area': 0,
             'Metropolitan cities': 0,
             'Others': 0}

    for i, data in g.nodes(data=True):

        region = data['name'].split(',')[1].strip().lower()

        area = classify_area(region)

        if area is not None:
            counts[area] += 1

    return counts


def calc_deg_by_region(net_type='global', direction='out'):

    assert (net_type in ['global', 'domestic'])
    assert (direction in ['in', 'out'])

    nets = construct_network(net_type=net_type)
    results = {}

    for iden in identifiers:

        g = nets[iden] 
        results[iden] = _calc_deg_by_region(g, direction)

    return results


def _calc_deg_by_region(g, direction):

    counts = {'Seoul': 0,
             'Capital area': 0,
             'Metropolitan cities': 0,
             'Others': 0}

    for src_id, dst_id, data in g.edges(data=True):

        if direction == 'out':
            name = g.nodes[src_id]['name']

        else:
            name = g.nodes[dst_id]['name']

        region = name.split(',')[1].strip().lower()

        area = classify_area(region)

        if area is not None:
            counts[area] += 1

    return counts


def grant_area_codes():

    from config.config import net_db_path

    db_name = 'KR_Integrated'

    conn = sql.connect(net_db_path(db_name))
    cursor = conn.cursor()

    id_to_code = {}

    nets = construct_network()

    for g in nets.values():
        _grant_area_codes(g, id_to_code)

    for id, code in id_to_code.items():

        cursor.execute(f"""
            UPDATE Nodes
            SET area_code = ?
            WHERE id = ?
        """, (code, id))

    conn.commit()
    conn.close()


def _grant_area_codes(g, id_to_code):

    for id, data in g.nodes(data=True):

        if data['nation'] != 'KR':
            continue

        address = data['address']

        if address is None:
            continue
        
        area_1 = address.split(',')[-1].strip().lower()
        area_2 = address.split(',')[-2].strip().lower()

        code_1 = area_codes.get(area_1)

        if code_1 is not None:
            id_to_code[id] = code_1
            continue

        code_2 = area_codes.get(area_2)

        if code_2 is not None:
            id_to_code[id] = code_2


def extract_for_cartogram(data, iden='Default'):

    import pandas as pd

    data_reg = {'AREA_CD': [],
                'AVG_RANK': [],
                'AVG_OUTDEG': [],
                'TOTAL_OUTDEG': [],
                'GYRAD_GEO': [],
                'GYRAD_RANK': []}

    for key, value in data.items():

        data_reg['AREA_CD'].append(key)
        data_reg['AVG_RANK'].append(value[0])
        data_reg['AVG_OUTDEG'].append(value[1])
        data_reg['TOTAL_OUTDEG'].append(value[2])
        data_reg['GYRAD_GEO'].append(value[3])
        data_reg['GYRAD_RANK'].append(value[4])

    for code in area_codes.values():

        if code not in data_reg['AREA_CD']:
            data_reg['AREA_CD'].append(code)
            data_reg['AVG_RANK'].append(max_domestic_ranks[iden.split('_')[1]] + 1)
            data_reg['AVG_OUTDEG'].append(0)
            data_reg['TOTAL_OUTDEG'].append(0)
            data_reg['GYRAD_GEO'].append(0)
            data_reg['GYRAD_RANK'].append(0)

    df = pd.DataFrame(data_reg)

    df.to_excel(xlsx_data_path(f'CARTO_{iden}'), index=False)


def extract_values_for_cartogram():

    nets = construct_network(net_type='domestic')

    for g in nets.values():
        mapping = _extract_values_for_cartogram(g)

        extract_for_cartogram(mapping, iden=g.name)


def _extract_values_for_cartogram(g):
    
    iden = g.name
    abbrev = qsi.get_abbrev(iden)

    code_to_rank = {}
    code_to_outdeg = {}
    code_to_gyrad_geo = {}
    code_to_gyrad_rank = {}

    id_to_gyrad_geo = _calc_radg(g, dist_type='geo')
    id_to_gyrad_rank = _calc_radg(g, dist_type='rank')

    for id, data in g.nodes(data=True):

        if data['nation'] != 'KR':
            continue

        rank = data[f'{abbrev}_rank_domestic']
        code = data['area_code']
        rad_geo = id_to_gyrad_geo.get(id)
        rad_rank = id_to_gyrad_rank.get(id)

        if all(item is not None for item in [rank, code, rad_geo, rad_rank]):

            if code not in code_to_rank:
                code_to_rank[code] = []
                code_to_outdeg[code] = []
                code_to_gyrad_geo[code] = []
                code_to_gyrad_rank[code] = []

            code_to_rank[code].append(rank)
            code_to_outdeg[code].append(g.out_degree(id))
            code_to_gyrad_geo[code].append(rad_geo)
            code_to_gyrad_rank[code].append(rad_rank)

    for code in code_to_rank.keys():

        avg_rank = np.mean(code_to_rank[code])
        avg_outdeg = np.mean(code_to_outdeg[code])
        total_outdeg = np.sum(code_to_outdeg[code])
        avg_gyrad_geo = np.mean(code_to_gyrad_geo[code])
        avg_gyrad_rank = np.mean(code_to_gyrad_rank[code])

        code_to_rank[code] = (avg_rank, avg_outdeg, total_outdeg, avg_gyrad_geo, avg_gyrad_rank)

    return code_to_rank


if __name__ == "__main__":

    # rs = [0, 1, 1, 2]
    # ms = [1, 1, 2, 3]

    # print("==== Ver 1 ====")
    # calc_radg(dist_type='geo')

    # print("==== Ver 2 ====")
    # calc_radg_ver2(dist_type='rank')

    extract_values_for_cartogram()

