import pandas as pd

import config.config as cfg


def integrate_geoinfo():
    
    geoidens = [f"GeoInfo_{iden.split('_')[1]}" for iden in cfg.identifiers]

    geoinfo_path = [cfg.csv_data_path(geoiden) for geoiden in geoidens]

    geoinfo_dict = {}

    for geo_path in geoinfo_path:

        df = pd.read_csv(geo_path)

        for id, row in df.iterrows():

            name = row['name']
            address = row['address']

            nation = row['nation']

            lat = row['latitude']
            lng = row['longitude']

            geoinfo_dict[name] = (nation, address, lat, lng)

        print(geoinfo_dict)

        data_list = [(name, *info) for name, info in geoinfo_dict.items()]

        df = pd.DataFrame(data_list, columns=['Name', 'Nation', 'Address', 'Latitude', 'Longitude'])
        df.sort_values(by='Nation')

        df.to_csv(cfg.csv_data_path("GeoInfo"), index=False)


def integrate_foreignnodes():
    
    nodeidens = [f"Nodes_{iden.split('_')[1]}" for iden in cfg.identifiers]

    nodeinfo_path = [cfg.csv_data_path(iden) for iden in nodeidens]

    nodeinfo_dict = {}

    for node_path in nodeinfo_path:

        df = pd.read_csv(node_path)

        for id, row in df.iterrows():

            name = row['name']

            nation = row['nation']

            if nation != 'KR':
                nodeinfo_dict[name] = (name, nation)

        data_list = [info for name, info in nodeinfo_dict.items()]

        df = pd.DataFrame(data_list, columns=['Name', 'Nation'])
        df.sort_values(by='Nation')

        df.to_csv(cfg.csv_data_path("ForeignNodes"), index=False)



if __name__ == "__main__":
    integrate_foreignnodes()


    