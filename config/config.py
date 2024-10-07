from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent

config_dir = root_dir / 'config'
data_dir = root_dir / 'data'
db_dir = root_dir / 'db'
fig_dir = root_dir / 'fig'

csv_data_dir = data_dir / 'csv'
xlsx_data_dir = data_dir / 'xlsx'

raw_db_dir = db_dir / 'raw'
net_db_dir = db_dir / 'net'

with open(config_dir / 'target.cfg') as f:
    identifiers = f.read().splitlines()


def csv_data_path(iden):
    return (csv_data_dir / iden).with_suffix('.csv')


def xlsx_data_path(iden):
    return (xlsx_data_dir / iden).with_suffix('.xlsx')


def raw_db_path(iden):
    return (raw_db_dir / iden).with_suffix('.db')


def net_db_path(iden):
    return (net_db_dir / iden).with_suffix('.db')


def fig_path(name, extension='.pdf'):
    return (fig_dir / name).with_suffix(extension)
