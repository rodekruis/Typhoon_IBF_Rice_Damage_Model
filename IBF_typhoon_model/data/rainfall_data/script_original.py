"""
Module to process the data for a typhoon that is required for feature building and
running the machine-learning pipeline.
There are a number of things that this module does:
(i) The methods download_gpm and download_srtm download the data for rainfall and elevation, respectively.
(ii) These data are then used, along with the windspeed and track data to obtain average values per municipality.
These data are then outputted into a CSV file for each typhoon that it is run for. It needs to be run
ONLY ONCE for each typhoon.
REQUIRED INPUTS:
(i) Typhoon name.
(ii) Start/End date for the typhoon.
(iii) IMERG data type: early, late or final: the data respository also needs to be checked, lest files are moved away
                        and the data doesn't exist anymore.
(iv) If the files need 'p-coding', then the p-code in the file needs to be specified (mostly deprecated now).
(v) Data-set file names for the municipalities, track and windspeed data files.
OUTPUTS:
    (i) A CSV of all the features, save the rainfall data.
    (ii) A CSV file of the rainfall data.
"""

import datetime as dt
import ftplib
import gzip
import os
import zipfile
from ftplib import FTP

import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from fiona.crs import from_epsg
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterstats import zonal_stats
from scipy import signal


def date_range(start_date, end_date): return [str(start_date + dt.timedelta(days=x))
                                              for x in range((end_date - start_date).days + 1)]


def unzip(zip_file, destination):
    os.makedirs(destination, exist_ok=True)

    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(destination)

    return


def reproject_file(gdf, file_name, force_epsg):

    print("Reprojecting %s to EPSG %i...\n" % (file_name, force_epsg), end="", flush=True)
    gdf = gdf.to_crs(epsg=force_epsg)

    return gdf


def reproject_raster(src_array, src_transform, src_epsg, dst_epsg, src_nodata=-32768, dst_nodata=-32768):
    """ Method to re-project a data-frame in the digital elevation model (DEM) to EPSG format.
    :param src_array: the data in DEM format
    :type src_array: pandas data-frame
    :param src_transform:
    :type src_transform:
    :param src_epsg:
    :type src_epsg:
    :param dst_epsg:
    :type dst_epsg:
    :param src_nodata:
    :type src_nodata:
    :param dst_nodata:
    :type dst_nodata:
    :raises:
    :returns:
    """
    src_height, src_width = src_array.shape
    dst_affine, dst_width, dst_height = calculate_default_transform(
        from_epsg(src_epsg),
        from_epsg(dst_epsg),
        src_width,
        src_height,
        *array_bounds(src_height, src_width, src_transform))

    dst_array = np.zeros((dst_width, dst_height))
    dst_array.fill(dst_nodata)

    reproject(
        src_array,
        dst_array,
        src_transform=src_transform,
        src_crs=from_epsg(src_epsg),
        dst_transform=dst_affine,
        dst_crs=from_epsg(dst_epsg),
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        resampling=Resampling.nearest)

    return dst_array, dst_affine


def slope(array, transform):
    """ Method to calculate the average slope of a district in degrees.
    :param array:
    :type array:
    :param transform:
    :type transform:
    :returns: slope_deg
    """
    height, width = array.shape
    bounds = array_bounds(height, width, transform)

    cellsize_x = (bounds[2] - bounds[0]) / width
    cellsize_y = (bounds[3] - bounds[1]) / height

    z = np.zeros((height + 2, width + 2))
    z[1:-1, 1:-1] = array
    dx = (z[1:-1, 2:] - z[1:-1, :-2]) / (2 * cellsize_x)
    dy = (z[2:, 1:-1] - z[:-2, 1:-1]) / (2 * cellsize_y)

    slope_deg = np.arctan(np.sqrt(dx * dx + dy * dy)) * (180 / np.pi)

    return slope_deg


def ruggedness(array):

    """ Method to calculate the average ruggedness of a district.
    :param array:
    :type array:
    :returns:
    :raises:
    """
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = kernel_x.transpose()
    dx = signal.convolve(array, kernel_x, mode='valid')
    dy = signal.convolve(array, kernel_y, mode='valid')

    tr_index = np.sqrt(dx ** 2 + dy ** 2)

    return tr_index


def download_gpm(start_date, end_date, download_path, type_imerg, force_download_early_data=False):

    """ Method that downloads gpm files.
    This method looks in the data repositories of NASA for rainfall data.
    :param start_date: A date object denoting the START date to search for rainfall data.
    :param end_date: A date object denoting the END date to search for rainfall data.
    :param download_path: A string denoting where the data should be downloaded to.
    :param type_imerg: Hart-coded st
    :param force_download_early_data: A bool switch to trigger forced download of early data.
        If this is FALSE, then the method looks in the normal place. If set to TRUE, it looks
        in the archive location. This is only done after checking whether the start and end dates
        cover the archive time (i.e., >= 01/12/2016).
    :returns: file_list, a  list of files?
    :raises: ftplib.allerrors
    """
    ppm_username = "bleumink@gmail.com"
    base_url = ""

    if type_imerg == "final":
        base_url = "arthurhou.pps.eosdis.nasa.gov"
        data_dir = "/pub/gpmdata/"  # data_dir/yyyy/mm/dd/gis
    elif type_imerg == "early":
        base_url = "jsimpson.pps.eosdis.nasa.gov"
        # Change directory to search if data has already been moved
        if force_download_early_data:
            data_dir = "/NRTPUB/imerg/gis/early/" + str(start_date.year) + "/" + str(start_date.month) + "/"
        else:
            data_dir = "/NRTPUB/imerg/gis/early/"  # all data from past 5 days is in this folder

    date_list = date_range(start_date, end_date)
    file_list = []

    os.makedirs(download_path, exist_ok=True)

    print("Connecting to: %s...\n" % base_url, end="", flush=True)

    with FTP(base_url) as ftp:

        try:
            ftp.login(ppm_username, ppm_username)
            print("Login OK...\n", end="", flush=True)
        except ftplib.all_errors as e:
            print("Error logging in:\n", e, end="", flush=True)

        for date in date_list:
            print(date)

            d, m, y = reversed(date.split('-'))
            day_path = os.path.join(download_path, y + m + d)
            os.makedirs(day_path, exist_ok=True)

            if type_imerg == "final":
                data_dir_final = "/pub/gpmdata/" +str(y) + '/' + str(m) + '/' + str(d) + '/gis' # data_dir/yyyy/mm/dd/gis          
				#ftp.cwd(os.path.join(data_dir, y, m, d, 'gis/'))
                ftp.cwd(data_dir_final)
            elif type_imerg == "early":
                ftp.cwd(data_dir)
            for entry in ftp.mlsd():

                file_name = entry[0]

                if (type_imerg == "final" and file_name.endswith(('tif', 'tfw')) and entry[0][3:6] == 'DAY') \
                        or (type_imerg == "early" and file_name.endswith(('30min.tif', '30min.tfw'))
                            and entry[0][23:31] == '%s%s%s' % (y, m, d)) \
                        or (type_imerg == "early" and file_name.endswith(('30min.tif.gz', '30min.tfw.gz'))
                            and entry[0][23:31] == '%s%s%s' % (y, m, d)):

                    file_path = os.path.join(day_path, file_name)

                    if file_name.endswith('tif'):
                        print("Retrieving %s...\n" % file_name, end="", flush=True)
                        file_list.append(file_path)
                        if os.path.isfile(file_path):
                            print("found locally...\n", end="", flush=True)
                    if not os.path.isfile(file_path):
                        with open(file_path, 'wb') as write_file:
                            ftp.retrbinary('RETR ' + file_name, write_file.write)
                    if file_name.endswith('tif.gz'):
                        new_file_name = os.path.splitext(file_path)[0]
                        with gzip.open(file_path, 'rb') as file_in:
                            file_data = file_in.read()
                            with open(new_file_name, 'wb') as file_out:
                                file_out.write(file_data)
                        file_list.append(new_file_name)
    return file_list


def download_srtm(bounding_box, download_path):

    """ Method to download srtm files?
    :param bounding_box:
    :type bounding_box:
    :param download_path:
    :type download_path:
    :return: tif_list, a list of tif files?
    :raises: IOErrior
    """
    base_url = "srtm.csi.cgiar.org"
    data_dir = "SRTM_V41/SRTM_Data_GeoTiff"

    tile_x0 = int((bounding_box[0] + 180) // 5) + 1
    tile_x1 = int((bounding_box[2] + 180) // 5) + 1
    tile_y0 = int((60 - bounding_box[3]) // 5) + 1
    tile_y1 = int((60 - bounding_box[1]) // 5) + 1

    tif_list = []
    zip_list = []
    ignore_list = []

    print("Checking local cache for SRTM tiles...\n", end="", flush=True)

    ignore_file = os.path.join(download_path, "ignore_tiles.txt")
    if os.path.isfile(ignore_file):
        with open(ignore_file, 'r') as file_to_open:
            for line in file_to_open.readlines():
                ignore_list.append(line.strip())

    for x_int in range(tile_x0, tile_x1 + 1):
        for y_int in range(tile_y0, tile_y1 + 1):
            if x_int > 9:
                x = str(x_int)
            else:
                x = "0" + str(x_int)
            if y_int > 9:
                y = str(y_int)
            else:
                y = "0" + str(y_int)

            tile_folder = os.path.join(download_path, "%s_%s" % (x, y))
            tile_path = os.path.join(tile_folder, "srtm_%s_%s.tif" % (x, y))
            zip_path = os.path.join(download_path, "srtm_%s_%s.zip" % (x, y))

            if os.path.isfile(tile_path):
                tif_list.append(tile_path)
            else:
                if "%s_%s" % (x, y) not in ignore_list:
                    zip_list.append((tile_folder, tile_path, zip_path, x, y))

    total_tiles = len(tif_list) + len(zip_list)
    print("found %i of %i tiles...\n" % (len(tif_list), total_tiles), end="", flush=True)

    if zip_list:
        print("Connecting to %s...\n" % base_url, end="", flush=True)
        with FTP(base_url) as ftp:
            ftp.login()
            print("OK!")
            ftp.cwd(data_dir)

            os.makedirs(download_path, exist_ok=True)

            for tile_folder, tile_path, zip_path, x, y in list(zip_list):

                zip_name = os.path.basename(zip_path)
                print("Retrieving %s...\n" % zip_name, end="", flush=True)

                if not os.path.isfile(zip_path):
                    with open(zip_path, 'wb') as write_file:
                        try:
                            ftp.retrbinary('RETR ' + zip_name, write_file.write)
                        except IOError:
                            print("skipped...\n", end="", flush=True)
                            os.remove(zip_path)
                            zip_list.remove((tile_folder, tile_path, zip_path, x, y))
                            ignore_list.append("%s_%s" % (x, y))

                else:
                    print("found locally...\n", end="", flush=True)

        if ignore_list:
            with open(ignore_file, 'w') as file_to_open:
                for tile in ignore_list:
                    file_to_open.write(tile + '\n')

    if zip_list:
        print("Unzipping downloaded tiles...\n", end="", flush=True)
        for tile_folder, tile_path, zip_path, x, y in zip_list:
            unzip(zip_path, tile_folder)
            tif_list.append(tile_path)

    return tif_list



def extract_coast(admin_geometry):
    """ Method to extract the coast length
    :param admin_geometry:
    :type admin_geometry:
    :returns: coast_length, cp_ratio tuple: the length of coast and the ratio of coast to ?
    :raises:
    """
    dissolve_poly = admin_geometry.unary_union
    coast_line = dissolve_poly.boundary
    coast_length = []
    cp_ratio = []

    for admin_shape in admin_geometry:
        perimeter = admin_shape.boundary
        intersect = perimeter.intersection(coast_line)
        ratio = intersect.length / perimeter.length
        coast_length.append(intersect.length)
        cp_ratio.append(ratio)

    return coast_length, cp_ratio


def cumulative_rainfall(admin_geometry, start_date, end_date, download_path, type_imerg):

    """ Method to calcualte the cumulative amount of rainfall from the typhoon.
    :param admin_geometry:
    :type admin_geometry:
    :param start_date: A date object signifying the start date of the typhoon.
    :param end_date: A date object signifying the end date of the typhoon.
    :param download_path: A string denoting where the data should be downloaded to.
    :param type_imerg: A string denoting the data file type: "early" or "final"
    :returns: sum_rainfall: A list of the total rainfall over the dates
    :raises:
    """

    file_list = download_gpm(start_date, end_date, download_path, type_imerg)
    if not file_list:
        force_download = True
        file_list = download_gpm(start_date, end_date, download_path, type_imerg, force_download)

    sum_rainfall = []

    if file_list:

        print("Reading GPM data...\n", end="", flush=True)

        raster_list = []
        transform = ''
        for input_raster in file_list:
            with rasterio.open(input_raster) as src:
                array = src.read(1)
                transform = src.transform  # src.affine <- affine is now deprecated
            array = np.ma.masked_where(array == 9999, array)
            raster_list.append(array)

        print("Calculating cumulative rainfall...\n", end="", flush=True)
        sum_raster = np.add.reduce(raster_list)
        sum_raster = sum_raster / 10
        if type_imerg == "final":
            sum_raster = sum_raster * 24

        sum_rainfall = zonal_stats(admin_geometry, sum_raster, stats='mean', nodata=-999, all_touched=True,
                                   affine=transform)
        sum_rainfall = [i['mean'] for i in sum_rainfall]

    else:
        print("No files were found/downloaded from the appropriate folder. Please investigate further.\n")
        pass

    return sum_rainfall


def srtm_features(admin_geometry, bounding_box, download_path, force_epsg):
    """ Method to extract srtm features.
    :param force_epsg:
    :param admin_geometry:
    :type admin_geometry:
    :param bounding_box:
    :type bounding_box:
    :param download_path:
    :type download_path:
    :returns: avg_elevation, avg_slope
    :raises:
    """

    file_list = download_srtm(bounding_box, download_path)

    print("Reading SRTM data...\n", end="", flush=True)

    raster_list = []

    for input_raster in file_list:
        raster_list.append(rasterio.open(input_raster))

    if len(raster_list) > 1:
        srtm_dem, srtm_transform = merge(raster_list, nodata=-32768)
        srtm_dem = srtm_dem[0]
    else:
        srtm_dem = raster_list[0].read(1)
        srtm_transform = raster_list[0].affine

    for input_raster in raster_list:
        input_raster.close()
    del raster_list

    print("Reprojecting DEM to EPSG %i...\n" % force_epsg, end="", flush=True)
    srtm_utm, transform_utm = reproject_raster(srtm_dem, srtm_transform, 4326, force_epsg, -32768, 0)

    print("Calculating mean elevation...\n", end="", flush=True)
    avg_elevation = zonal_stats(admin_geometry, srtm_utm, stats='mean', nodata=-32768, all_touched=True,
                                affine=transform_utm)
    avg_elevation = [i['mean'] for i in avg_elevation]

    print("Calculating mean slope...\n", end="", flush=True)
    avg_slope = zonal_stats(admin_geometry, slope(srtm_utm, transform_utm), stats='mean', nodata=0, all_touched=True,
                            affine=transform_utm)
    avg_slope = [i['mean'] for i in avg_slope]

    return avg_elevation, avg_slope


def process_tyhoon_data(typhoon_to_process, typhoon_name):
    """ Method to process the data for typhoons.
    :param typhoon_to_process: A dictionary instance containing all required information about the data
            for the typhoon.
    :param typhoon_name: The name of the typhoon (can be just passed through as the dictionary key.
    """
	
    typhoon = typhoon_to_process.get('typhoon')
    #windspeed_source = typhoon_to_process.get('windspeed_source')
    #windspeed_type = typhoon_to_process.get('windspeed_type')
    subfolder = typhoon + "/" #+ windspeed_source + "/" + windspeed_type + "/"
	
    current_path = os.path.dirname(os.path.abspath(__file__))

    # Specify location of datasets
    workspace = os.path.abspath(os.path.join(current_path, "..", "./data/input/" + subfolder))

    # Start/End date for precipitation data
    start_date = typhoon_to_process.get('dates')[0]
    end_date = typhoon_to_process.get('dates')[1]

    print("start_date is:", start_date, "end date of typhoon is:", end_date)

    # IMERG data type, either "early" (6hr), "late" (18hr) or "final" (4 months),
    # see https://pps.gsfc.nasa.gov/Documents/README.GIS.pdf
    imerg_type = typhoon_to_process.get('imerg_type')  # "early"
    print("imerg_type:", imerg_type)

    # Specify P Coded column in administrative boundaries file
    p_code = "Mun_Code"

    windspeed_file_name = typhoon_to_process.get('windspeed_file_name')
    print("Using a windspeed file name of:", windspeed_file_name)

    track_file_name = typhoon_to_process.get('track_file_name')
    print("Using track file of:", track_file_name)

    # Specify output file names
    # output_shp_name = typhoon_name + "_matrix.shp"
    output_matrix_csv_name = typhoon_name + "_matrix.csv"
    #output_rainfall_csv_name = typhoon_name + "_rainfall.csv"

    # Output will be in this CRS, datasets are reprojected if necessary
    force_epsg = 32651  # UTM Zone 51N

    t0 = dt.datetime.now()

    gpm_path = os.path.join(workspace, "GPM")
    #  srtm_path = os.path.join(workspace, "SRTM")

    #admin_file = os.path.join(workspace, admin_file_name)
    windspeed_file = os.path.join(workspace, windspeed_file_name)
    track_file = os.path.join(workspace, track_file_name)

    # output_shp_file = os.path.join(workspace, output_shp_name)
    output_matrix_csv_file = os.path.join(workspace, output_matrix_csv_name)
    #output_rainfall_csv_file = os.path.join(workspace, output_rainfall_csv_name)

    # Loading shapefiles
    print("Importing shapefiles...\n", end="", flush=True)
    windspeed_gdf = gpd.GeoDataFrame()
    track_gdf = gpd.GeoDataFrame()

    try:
        windspeed_gdf = gpd.GeoDataFrame.from_file(windspeed_file)
    except IOError as ioe:
        print("Could not load file properly", ioe, end="", flush=True)
    try:
        track_gdf = gpd.GeoDataFrame.from_file(track_file)
    except IOError as ioe:
        print("Could not load file properly", ioe, end="", flush=True)

    # Check if CRS is defined and default to WGS 84 if not
    if not windspeed_gdf.crs:
        windspeed_gdf.crs = from_epsg(4326)
    if not track_gdf.crs:
        track_gdf.crs = from_epsg(4326)

	# Check CRS of each layer and reproject if necessary
    if int(windspeed_gdf.crs['init'].split(':')[1]) != force_epsg:
        windspeed_gdf = reproject_file(windspeed_gdf, windspeed_file_name, force_epsg)

    if int(track_gdf.crs['init'].split(':')[1]) != force_epsg:
        track_gdf = reproject_file(track_gdf, track_file_name, force_epsg)

    output_columns = [
        'avg_speed_mph',
        'distance_typhoon_km',
        'area_km2',
        'Mun_Code',
        'OBJECTID',
        'x_pos',
        'y_pos',
		'Rainfall',
        'geometry']

    #output_columns_rainfall = [
    #    'M_Code',
    #    'Rainfallme',
    #    'geometry']

    output_gdf = gpd.GeoDataFrame(columns=output_columns, crs=from_epsg(force_epsg))
    #output_gdf_rainfall = gpd.GeoDataFrame(columns=output_columns_rainfall, crs=from_epsg(force_epsg))

    print("Assigning P codes...\n", end="", flush=True)
    output_gdf['Mun_Code'] = admin_gdf[p_code]
    #output_gdf_rainfall['M_Code'] = admin_gdf[p_code]

    print("Calculating average windspeeds...\n", end="", flush=True)
    #output_gdf['avg_speed_mph'] = average_windspeed(admin_gdf.geometry,windspeed_gdf.geometry, windspeed_gdf['Name'],admin_gdf[p_code])
    output_gdf['avg_speed_mph'] = \
        admin_gdf.geometry.apply(average_windspeed, args=(windspeed_gdf.geometry, windspeed_gdf['Name']))

    print("Calculating centroid distances...\n", end="", flush=True)
    output_gdf['distance_typhoon_km'] = \
        admin_gdf.centroid.geometry.apply(lambda g: track_gdf.distance(g).min()) / 10 ** 3
    output_gdf['x_pos'] = admin_gdf.centroid.map(lambda p: p.x)
    output_gdf['y_pos'] = admin_gdf.centroid.map(lambda p: p.y)

    print("Calculating areas...\n", end="", flush=True)
    output_gdf['area_km2'] = admin_gdf.area / 10 ** 6

    # Calculating cumulative rainfall
    if not imerg_type == 'trmm':
        output_gdf['Rainfall'] = cumulative_rainfall(admin_geometry_wgs84, start_date, end_date, gpm_path, imerg_type)
	#output_gdf_rainfall['Rainfallme'] = \
    #    cumulative_rainfall(admin_geometry_wgs84, start_date, end_date, gpm_path, imerg_type)

    # Assigning geometry
    output_gdf.geometry = admin_gdf.geometry
    #output_gdf_rainfall.geometry = admin_gdf.geometry

    # TODO: move the rainfall data into the other CSV file.
    if output_matrix_csv_name:
        print("Exporting output to %s...\n" % output_matrix_csv_name, end="", flush=True)
        output_df = output_gdf.drop('geometry', axis=1)
        output_df.to_csv(output_matrix_csv_file)

        #print("Exporting output to %s...\n" % output_rainfall_csv_name, end="", flush=True)
        #output_df_rainfall = output_gdf_rainfall.drop('geometry', axis=1)
        #output_df_rainfall.to_csv(output_rainfall_csv_file)

    t_total = dt.datetime.now()
    print('Completed in %fs\n' % (t_total - t0).total_seconds(), end="", flush=True)


    
##########################
### FILL IN INPUT HERE ###
##########################

#typhoons = ['Nock-Ten','Hagupit','Haima','Haiyan','Melor','Rammasun','Bopha','Goni','Kalmaegi','Koppu','Matmo','Sarika','Soudelor','Utor'] 
typhoons = ['Nock-Ten','Hagupit','Haima','Haiyan','Melor','Rammasun','Bopha','Goni','Kalmaegi','Koppu','Matmo','Sarika','Soudelor','Utor'] 
#windspeed_file_name = 
#track_file_name = ...

##########################
### FILL IN INPUT HERE ###
##########################

# Start of actual Code
current_path = os.path.dirname(os.path.abspath(__file__))
workspace_admin = os.path.abspath(os.path.join(current_path, "..", "./data/input/"))
admin_file_name = "PHL_adm3_PSA_pn_2016June/PHL_adm3_PSA_pn_2016June_mapshaper.shp"

typhoon_metadata_filename = os.path.join(workspace_admin, 'metadata_typhoons.csv')
typhoon_metadata = pd.read_csv(typhoon_metadata_filename,delimiter=';')
typhoon_metadata = typhoon_metadata.set_index('typhoon').to_dict()

typhoons_dict = dict()

i=0
for typhoon in typhoons:
    #if i==0:
        case = typhoon
        typhoons_dict[case] = {
            "typhoon": typhoon,
            "dates": [dt.datetime.strptime(typhoon_metadata['startdate'][typhoon],'%d-%m-%Y').date(), dt.datetime.strptime(typhoon_metadata['enddate'][typhoon],'%d-%m-%Y').date()], 
            "imerg_type": typhoon_metadata['imerg_type'][typhoon],
            "windspeed_file_name": 'JTWC\inc_translational\\' + typhoon + '_JTWC_inc_translational_windspeed.shp',
            "track_file_name": 'JTWC\inc_translational\\' + typhoon + '_JTWC_inc_translational_track.shp',
            }
        i=i+1
				
		

		
# Loading admin-files only once instead of every time
current_path = os.path.dirname(os.path.abspath(__file__))
workspace_admin = os.path.abspath(os.path.join(current_path, "..", "./input_data/"))
admin_file = os.path.join(workspace_admin, admin_file_name)
print("Importing adminfile...\n", end="", flush=True)
admin_gdf = gpd.GeoDataFrame()
try:
	admin_gdf = gpd.GeoDataFrame.from_file(admin_file)
except IOError as ioe:
	print("Could not load file properly", ioe, end="", flush=True)
# Check if CRS is defined and default to WGS 84 if not
if not admin_gdf.crs:
	admin_gdf.crs = from_epsg(4326)
# Keeping an unprojected copy of admin area geometry in WGS84 to speed up raster calculations
if int(admin_gdf.crs['init'].split(':')[1]) != 4326:
	admin_geometry_wgs84 = reproject_file(admin_gdf.geometry, admin_file_name, 4326)
else:
	admin_geometry_wgs84 = admin_gdf.geometry
# Output will be in this CRS, datasets are reprojected if necessary
	force_epsg = 32651  # UTM Zone 51N
# Check CRS of each layer and reproject if necessary
if int(admin_gdf.crs['init'].split(':')[1]) != force_epsg:
	admin_gdf = reproject_file(admin_gdf, admin_file_name, force_epsg)


	
#Start loop to process typhoon-data	
for key in typhoons_dict:
    print("Processing typhoon data for:", key)
    process_tyhoon_data(typhoons_dict[key], key)