# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 07:42:18 2025

@author: Magnolia
"""


import h5py as h5
import pandas as pd
import geopandas as gpd
import numpy as np

from pathlib import Path

from tqdm import tqdm
import pickle
import subprocess

from shapely.geometry import Point

from atmoz.resources.timeConversions import h5Dataset_timestamp

import pickle

# Should be moved to an lookups directory of atmoz or something, but this is the ICARTT header structure as of 2024.
ict_header = {
    "number_of_lines": (
        "Number of lines in header, file format index "
        "(most files use 1001) - comma delimited. "
        ),
    "pi_Name": "PI last name, first name/initial. ",
    "organization": "Organization/affiliation of PI ",
    "data_source_description": (
        "Data source description (e.g., instrument name, platform name, "
        "model name, etc.). "
        ),
    "mission_name": "Mission name (usually the mission acronym). ",
    "file_volumne_number": (
        "File volume number,  number of file volumes (these integer values "
        "are used when the data require more than one file per day; for data "
        "that require only one file these values are set to 1, 1) - comma delimited. "
        ),
    "datetime_start": (
        "UTC date when data begin, UTC date of data reduction or revision "
        "- comma delimited (yyyy, mm, dd, yyyy, mm, dd). "
        ),
    "data_interval": (
        "Data Interval (This value describes the time spacing (in seconds) "
        "between consecutive data records.  It is the (constant) interval "
        "between values of the independent variable. For 1 Hz data the data "
        "interval value is 1 and for 10 Hz data the value is 0.1.  All intervals "
        "longer than 1 second must be reported as Start and Stop times, and the "
        "Data Interval value is set to 0.  The Mid-point time is required when "
        "it is not at the average of Start and Stop times. For additional "
        "information see Section 2.5 below.). "
        ),
    "independent_variable": (
        "Description or name of independent variable (This is the name chosen "
        "for the start time. It always refers to the number of seconds UTC from "
        "the start of the day on which measurements began.  It should be noted "
        "here that the independent variable should monotonically increase even "
        "when crossing over to a second day.). "
        ),
    "number_of_variables": (
        "Number of variables (Integer value showing the number of dependent "
        "variables: the total number of columns of data is this value plus one.). "
        ),
    "scale_factors": (
        "Scale factors (1 for most cases, except where grossly inconvenient) "
        "- comma delimited. "
        ),
    "missing_data_indicator": (
        "Missing data indicators (This is -9999 (or -99999, etc.) for any "
        "missing data condition, except for the main time (independent) "
        "variable which is never missing) - comma delimited. "
        ),
    "variable_names_and_units": (
        "Variable names and units (Short variable name and units are required, "
        "and optional long descriptive name, in that order, and separated by "
        "commas. If the variable is unitless, enter the keyword 'none' for its "
        "units. Each short variable name and units (and optional long name) are "
        "entered on one line. The short variable name must correspond exactly to "
        "the name used for that variable as a column header, i.e., the last "
        "header line prior to start of data.). "
        ),
    "number_of_special_comments": (
        "Number of SPECIAL comment lines (Integer value indicating the number "
        "of lines of special comments, NOT including this line.)"
        ),
    "special_comments": (
        "Special comments (Notes of problems or special circumstances unique to "
        "this file. An example would be comments/problems associated with a "
        "particular flight.)"
        ),
    "number_of_normal_comments": (
        "Number of Normal comments (i.e., number of additional lines of SUPPORTING "
        "information: Integer value indicating the number of lines of additional "
        "information, NOT including this line.). "
        ),
    "normal_comments": (
        "Normal comments (SUPPORTING information: This is the place for "
        "investigators to more completely describe the data and measurement "
        "parameters. The supporting information structure is described below "
        "as a list of key word: value pairs. Specifically include here "
        "information on the platform used, the geo-location of data, measurement "
        "technique, and data revision comments. Note the non-optional information "
        "regarding uncertainty, the upper limit of detection (ULOD) and the lower "
        "limit of detection (LLOD) for each measured variable. The ULOD and LLOD "
        "are the values, in the same units as the measurements that correspond "
        "to the flags -7777’s and -8888’s within the data, respectively. The "
        "last line of this section should contain all the “short” variable names "
        "on one line. The key words in this section are written in BOLD below "
        "and must appear in this section of the header along with the relevant "
        "data listed after the colon. For key words where information is not "
        "needed or applicable, simply enter N/A.). "
        ),
    "PI_CONTACT_INFO": (
        "PI_CONTACT_INFO: Phone number, mailing address, and email address "
        "and/or fax number. "
        ),
    "PLATFORM": (
        "Platform or site information. "
        ),
    "LOCATION": (
        "including lat/lon/elev if applicable. "
        ),
    "ASSOCIATED_DATA":(
        "File names with associated data: location data, aircraft parameters, "
        "ship data, etc. "
        ),
    "INSTRUMENT_INFO": (
        "Instrument description, sampling technique and peculiarities, "
        "literature references, etc. "
        ),
    "DATA_INFO": (
        "Units and other information regarding data manipulation."
        ),
    "UNCERTAINTY": (
        "Uncertainty information, whether a constant value or function, if the "
        "uncertainty is not given as separate variables. "
        ),
    "ULOD_FLAG": (
        "-7777 (Upper LOD flag, always -7’s). "
        ),
    "ULOD_VALUE": (
        " Upper LOD value (or function) corresponding to the -7777’s flag in "
        "the data records. "
        ),
    "LLOD_FLAG": (
        "-8888 (Lower LOD flag, always -8’s). "
        ),
    "LLOD_VALUE": (
        "Lower LOD value (or function) corresponding to the -8888’s flag in "
        "the data records. "
        ),
    "DM_CONTACT_INFO": (
        "Data Manager -- Name, affiliation, phone number, mailing address, "
        "email address and/or fax number."
        ),
    "PROJECT_INFO": (
        "Study start & stop dates, web links, etc. "
        ),
    "STIPULATIONS_ON_USE": (
        "self explanatory. "
        ),
    "OTHER_COMMENTS": (
        "Any other relevant information"
        ),
    "REVISION": (
        "R# See file names discussion"
        ),
    "R#": (
        "comments specific to this data revision. The revision numbers and the "
        "associated comments are cumulative in the data file. This is required "
        "in order to track the changes that have occurred to the data over time. "
        "Pre-pend the information to this section so that the latest revision "
        "number and comments always start this part of the header information. "
        "The latest revision data should correspond to the revision date on "
        "Line 7 of the main file header. "
        ),
    "ICARTT_DATA_FORMAT": r"https://www-air.larc.nasa.gov/missions/etc/IcarttDataFormat.htm"
    }

# Utilities
def read_h5(filepath):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    def recursive_load(h5_obj):
        """Recursively walk through the HDF5 structure and return nested dicts."""
        result = {}
        for key, item in h5_obj.items():
            if isinstance(item, h5.Dataset):
                # Convert dataset to a NumPy array
                result[key] = item[()]
            elif isinstance(item, h5.Group):
                # Recurse into subgroups
                result[key] = recursive_load(item)
        # Add attributes if present
        if h5_obj.attrs:
            result["_attrs"] = {k: v for k, v in h5_obj.attrs.items()}
        return result

    with h5.File(filepath, "r") as f:
        data = recursive_load(f)

    return data, filepath.name

# This should be moved to atmoz.resources or something, but this is a wrapper for h5py datasets that allows lazy loading and full NumPy-like behavior.
class H5Dataset:
    """Wrap h5py.Dataset for lazy loading with full NumPy-like behavior."""
    def __init__(self, dataset: h5.Dataset):
        self._dataset = dataset

    def __array__(self, dtype=None, **kwargs):
        arr = self._dataset[()]
        if dtype is not None:
            arr = arr.astype(dtype, copy=kwargs.get("copy", True))
        return arr

    # Arithmetic operators
    def __mul__(self, other):
        return np.array(self) * other

    def __rmul__(self, other):
        return other * np.array(self)

    def __add__(self, other):
        return np.array(self) + other

    def __radd__(self, other):
        return other + np.array(self)

    def __sub__(self, other):
        return np.array(self) - other

    def __rsub__(self, other):
        return other - np.array(self)

    def __truediv__(self, other):
        return np.array(self) / other

    def __rtruediv__(self, other):
        return other / np.array(self)

    # Indexing/slicing
    def __getitem__(self, key):
        return self._dataset[key]

    def __len__(self):
        return len(self._dataset)

    @property
    def shape(self):
        return self._dataset.shape

    @property
    def dtype(self):
        return self._dataset.dtype

    @property
    def attrs(self):
        return self._dataset.attrs

    def __repr__(self):
        return f"<H5Dataset {self._dataset.name} shape={self._dataset.shape} dtype={self._dataset.dtype}>"


# This should be moved to atmoz.resources or something, but this is a wrapper for h5py groups that allows dot-access, tab completion, and lazy dataset access.
class H5Node:
    """Wrap h5py.Group to allow dot-access, tab completion, and lazy dataset access."""
    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, key):
        item = self._obj[key]
        if isinstance(item, h5.Group):
            return H5Node(item)
        elif isinstance(item, h5.Dataset):
            return H5Dataset(item)
        return item

    def __getattr__(self, name):
        if name in self._obj:
            return self[name]
        if name == "attrs":
            return self._obj.attrs
        if name == "keys":
            return self.keys
        raise AttributeError(f"{name} not found in HDF5 group '{self._obj.name}'")

    def keys(self):
        return list(self._obj.keys())

    def __dir__(self):
        # Only include child **groups** for tab completion
        # group_keys = [k for k in self._obj.keys() if isinstance(self._obj[k], h5.Group)]
        return list(self._obj.keys()) + list(super().__dir__())

    def __repr__(self):
        cls = type(self._obj).__name__
        return f"<H5Node {cls} '{self._obj.name}'>"


# This should be moved to atmoz.resources or something, but this is a utility function to read an HDF5 file and return a wrapped H5Node for lazy access.
def read_h5_lazy(filepath):
    filepath = Path(filepath)
    f = h5.File(filepath, "r", swmr=True)
    return H5Node(f), filepath.name


class HSRL2:
    def __init__(self):
        self.data = {}

    def import_data(self, filepaths):
        filepaths = [Path(f) for f in filepaths]
        for filepath in tqdm(filepaths, desc="Registering HDF5 handles", ncols=80):
            node, filename = read_h5_lazy(filepath)
            self.data[filename] = node

    def close_all(self):
        """Close all open files safely."""
        for node in self.data.values():
            if isinstance(node._obj, h5.File):
                node._obj.close()

class TOLNet:
    def __init__(self):
        self.data = {}

    def import_data(self, filepaths):
        filepaths = [Path(f) for f in filepaths]
        for filepath in tqdm(filepaths, desc="Registering HDF5 handles", ncols=80):
            node, filename = read_h5_lazy(filepath)
            self.data[filename] = node

    def close_all(self):
        """Close all open files safely."""
        for node in self.data.values():
            if isinstance(node._obj, h5.File):
                node._obj.close()

# This should be moved to atmoz.resources or something, but this is a utility function to batch convert HDF4 files to HDF5 using the h4toh5 tool from the HDF Group.
def convert_hdf4_to_hdf5(h4_dir, h5_dir, h4toh5_exe=r"C:\Program Files\HDF_Group\H4H5\2.2.5\bin\h4toh5convert.exe"):
    """
    Batch convert all HDF4 (.hdf) files in a directory to HDF5 (.h5) using h4toh5 tool.

    Parameters:
        h4_dir (str or Path): Folder containing HDF4 files.
        h5_dir (str or Path): Folder where converted HDF5 files will be saved.
        h4toh5_exe (str or Path): Full path to the h4toh5 executable.

    Returns:
        List[Path]: Paths to the converted HDF5 files.
    """
    h4_dir = Path(h4_dir)
    h5_dir = Path(h5_dir)
    h5_dir.mkdir(parents=True, exist_ok=True)

    converted_files = []

    hdf_files = list(h4_dir.glob("*.hdf"))
    if not hdf_files:
        print(f"No HDF4 files found in {h4_dir}")
        return converted_files

    for hdf_file in hdf_files:
        output_file = h5_dir / (hdf_file.stem + ".h5")
        print(f"Converting {hdf_file} -> {output_file}")

        try:
            subprocess.run([str(h4toh5_exe), str(hdf_file), str(output_file)],
                           check=True)
            print(f"✅ Conversion successful: {output_file}")
            converted_files.append(output_file)
        except subprocess.CalledProcessError:
            print(f"❌ Conversion failed: {hdf_file}")

    print("All files processed.")
    return converted_files

from charset_normalizer import from_path

def detect_encoding(filepath):
    result = from_path(filepath).best()
    if result is not None:
        return result.encoding
    return None

def read_ict(filepaths):
    if not isinstance(filepaths, list):
        filepaths = list(filepaths)

    filepaths = [Path(f) for f in filepaths]

    data = {}
    for filepath in filepaths:
        encoding = detect_encoding(filepath)
        # metadata = {}
        with open(filepath, "r") as f:
            line_1 = f.readline()
            start = int(line_1.split(",")[0]) - 1
            header = [
                line_1
                if i == 0
                else f.readline()
                for i in range(0, start+1)
                ]

        column_names = [
            x.replace(" ", "")
            for x in header[-1].rstrip("\n").split(", ")
            ]

        temp = pd.read_csv(
            filepath,
            skiprows=start+1,
            low_memory=False,
            encoding=encoding,
            names = column_names
            )

        start_day = "-".join(header[6].split(", ")[:3])

        temp["Datetime"] = temp["Seconds_UTC"].apply(
            lambda x: pd.Timedelta(x, unit="s")
            + pd.Timestamp(start_day, tz="UTC")
            )

        temp.drop(columns=["Index_number", "Seconds_UTC"], inplace=True)
        temp.set_index("Datetime", inplace=True)

        data[filepath.name] = {
            "header": header,
            "data": temp
            }

        # Trying to Dynamic Read ICARTT Header - This is hard.
        # header_keys = list(ict_header.keys())
        # metadata = {
        #     key: header[i].split(", ")
        #     for i, key in zip(range(12), header_keys)
        #     }

        # num_vars = int(metadata["number_of_variables"][0])
        # variables_info = [x.split() for x in header[12:num_vars]]
        # metadata[header_keys[12]] = variables_info

        # print(len(header_keys))
        # for i in range( 13 + num_vars, len(header) ):
        #     print(i)
        #     print(header_keys[i-num_vars])
        #     print(header[i])
        #     metadata[header_keys[i-num_vars]] = header[i]




        # start, _ = contents
        # for key in ict_header.keys():
        #     metadata[key] =
        # start, _ = f.readline().split(",")
        # start = int(start) - 2
        # for i in range(1, start):
        #     keys = list(ict_header.keys())
        #     line = f.readline()
        #     if i == 9:
        #         num_vars = int(line)
        #         print(num_vars)
        #         variables = []
        #     elif i > 12 and i < (12 + num_vars):
        #         variables.append(line)

        #     metadata[keys[i]] = line
        # print(variables)


    return data

def to_df(array, timestamps, altitude):
    if array.ndim == 1:
        if len(timestamps) == len(array):
            array = array[:, np.newaxis]  # make it (N,1)
        elif len(altitude) == len(array):
            array = array[np.newaxis, :]  # make it (1,N)
        else:
            raise ValueError(f"Cannot align 1D array in dataset")

    if array.shape[0] == len(timestamps) and array.shape[1] == len(altitude):
        df = pd.DataFrame(array, columns=altitude, index=timestamps)
    elif array.shape[1] == len(timestamps) and array.shape[0] == len(altitude):
        df = pd.DataFrame(array.T, columns=altitude, index=timestamps)
    else:
        raise ValueError(f"Shape mismatch in dataset: array {array.shape}, timestamps {len(timestamps)}, altitudes {len(altitude)}")
    return df
    
def main_import(dir_path: Path): 
    if not isinstance(dir_path, Path): 
        dir_path = Path(dir_path)
    
    # - HSRL2 - #
    data_filepath = r"HSRL2.pickle"
    if not (dir_path / data_filepath).is_file():
        hsrl2_data = HSRL2()
        hsrl2_data.import_data( list( (dir_path / r"HSRL2").glob("*R1.h5") ) )

        hsrl2 = {}
        for key, dataset in hsrl2_data.data.items():
            z = dataset.z[()]
            lat = dataset.lat[()]
            lon = dataset.lon[()]

            geometry = [Point(xy) for xy in zip(lon.flatten(), lat.flatten())]
            timestamps = h5Dataset_timestamp(dataset.time)

            ozone_array = dataset.DataProducts.O3[()]

            ozone_df = to_df(ozone_array, timestamps, z)
            units = dataset.DataProducts.O3.attrs.get("units", "")
            hsrl2[key] = gpd.GeoDataFrame(ozone_df, geometry=geometry, crs="EPSG:4326")

        with open(dir_path / data_filepath, "wb") as f:
            pickle.dump(hsrl2, f)
    else: 
        with open(dir_path / data_filepath, "rb") as f:
            hsrl2 = pickle.load(f)

    # - TOLNet - #
    data_filepath = r"TOLNet.pickle"
    if not (dir_path / data_filepath).is_file():
        tolnet_data = TOLNet()
        tolnet_data.import_data( list( (dir_path / r"TOLNet_hdf5").glob("*.h5") ) )

        tolnet = {}
        for key, dataset in tolnet_data.data.items():

            z = dataset['ALTITUDE'][()].astype(float)
            lat = dataset['LATITUDE.INSTRUMENT'][()].astype(float)
            lon = dataset['LONGITUDE.INSTRUMENT'][()].astype(float)

            timestamps = h5Dataset_timestamp(dataset['DATETIME.START'])
            geometry = [Point(xy) for xy in zip(lon, lat)] * len(timestamps)

            ozone_array = dataset['O3.MIXING.RATIO.VOLUME_DERIVED'][()].astype(float)
            # uncertainty_array = dataset['O3.MIXING.RATIO.VOLUME_DERIVED_UNCERTAINTY.RANDOM.STANDARD'].astype(float)
            # uncertainty_array[uncertainty_array <= -999] = np.nan
            ozone_array[ozone_array <= -999] = np.nan 

            # units = dataset['O3.MIXING.RATIO.VOLUME_DERIVED'].attrs.get("units", "")
            # tolnet[key] = {
            #     "values": gpd.GeoDataFrame(ozone_df, geometry=geometry, crs="EPSG:4326"),
            #     "uncertainty": gpd.GeoDataFrame(uncertainty_df, geometry=geometry, crs="EPSG:4326"),
            #     }
            
            ozone_df = to_df(ozone_array, timestamps, z)

            tolnet[key] = gpd.GeoDataFrame(ozone_df, geometry=geometry, crs="EPSG:4326")

        with open(dir_path / data_filepath, "wb") as f:
            pickle.dump(tolnet, f)
    else: 
        with open(dir_path / data_filepath, "rb") as f:
            tolnet = pickle.load(f)



    # - Sondes - #
    data_filepath = r"sondes.pickle"
    if not (dir_path / data_filepath).is_file(): 
        sondes_dict = read_ict(Path(r"./data/Sondes").glob("*.ict"))

        concat = pd.concat({file: x["data"] for file, x in sondes_dict.items()})
        concat.index.names = ["filename", "timestamp"]

        concat = concat.mask(concat < -999, np.nan)

        concat.dropna(subset=["Latitude_deg", "Longitude_deg"], inplace=True)

        concat["geometry"] = [Point(xy) for xy in zip(concat["Longitude_deg"], concat["Latitude_deg"])]
        sondes = gpd.GeoDataFrame(concat, geometry="geometry", crs="EPSG:4326")

        with open(dir_path / data_filepath, "wb") as f: 
            pickle.dump(sondes, f)

    else: 
        with open(dir_path / data_filepath, "rb") as f: 
            sondes = pickle.load(f)
            
    return {
        "hsrl2": hsrl2, 
        "tolnet": tolnet,
        "sondes": sondes
    }

def convert_asos(df): 
    temp = df.copy() 
    temp["datetime"] = pd.to_datetime(temp["valid"], utc=True)

    geometry = temp[["station", "lat", "lon"]].copy().drop_duplicates(keep='first')
    geometry["geometry"] = [Point(xy) for xy in zip(geometry['lon'], geometry['lat'])]
    geometry = gpd.GeoDataFrame(geometry, geometry='geometry', crs="EPSG:4326")

    temp.set_index(["station", "datetime"], inplace=True)
    temp = temp[[
                "lon", 
                "lat", 
                "relh", 
                "elevation", 
                "dwpf", 
                "tmpf", 
                "sknt", 
                "tmpf", 
                "alti", 
                "vsby", 
                "skyl1", 
                "skyl2", 
                "skyl3",
                "skyl4"
                ]]

    result = (
        temp
        .groupby(level=["station"])
        .resample("1 h", level="datetime")
        .mean() 
        )
    return {
        "data": result, 
        "geometry": geometry
    }

if __name__ == "__main__":
    print("Ran as Main")
    # h4_folder = r"E:\NPP\STAQS\data\TOLNet_hdf4"
    # h5_folder = r"E:\NPP\STAQS\data\TOLNet_hdf5"
    # converted = convert_hdf4_to_hdf5(h4_folder, h5_folder)

    # data_dir = Path("./data/TOLNet_hdf5")
    # files = list(data_dir.glob("*.h5"))

    # tolnet = TOLNet()
    # tolnet.import_data(files[:2])

    # data_dir = Path("./data/HSRL2")
    # files = list(data_dir.glob("*R1.h5"))

    # hsrl2 = HSRL2()
    # hsrl2.import_data(files[:2])

    # data_dir = Path(r"./data/Sondes")
    # data = read_ict(data_dir.glob("*.ict"))









