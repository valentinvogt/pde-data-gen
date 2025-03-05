import netCDF4 as nc
import argparse

def read_file(filename):
    with nc.Dataset(filename, 'r') as root:
        print(root)

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="Path to netcdf file")
args = parser.parse_args()

read_file(args.filename)
