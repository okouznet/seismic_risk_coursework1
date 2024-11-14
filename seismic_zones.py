# create tiff file layers
import pandas as pd
from PIL.ImageChops import overlay
from dotmap import DotMap
import contextily as cx
import glob

from networkx.classes import add_path

import util.catalogue_processing as catalogue_processing
import geopandas as gpd
import config as config
import ee
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import geemap
from osgeo import ogr, gdal
from config import OLYMPIA_GEO



def rasterize_shapefile(InputVector, OutputImage, RefImage):
    # add elevation, soil, projected faults, eearthquakes
    input_shapefile = gpd.read_file(InputVector).to_crs("EPSG:4326")
    input_shapefile.to_file(InputVector, driver='ESRI Shapefile')
    # first convert InputVector to needed CRS

    gdalformat = 'GTiff'
    datatype = gdal.GDT_Byte
    burnVal = 1  # value for the output image pixels
    ##########################################################
    # Get projection info from reference image
    Image = gdal.Open(RefImage, gdal.GA_ReadOnly)

    # Open Shapefile
    Shapefile = ogr.Open(InputVector)
    Shapefile_layer = Shapefile.GetLayer()

    # Rasterise
    print("Rasterising shapefile...")
    Output = gdal.GetDriverByName(gdalformat).Create(OutputImage, Image.RasterXSize, Image.RasterYSize, 1, datatype,
                                                     options=['COMPRESS=DEFLATE'])
    Output.SetProjection(Image.GetProjectionRef())
    Output.SetGeoTransform(Image.GetGeoTransform())

    # Write data to band 1
    Band = Output.GetRasterBand(1)
    Band.SetNoDataValue(0)
    gdal.RasterizeLayer(Output, [1], Shapefile_layer, burn_values=[burnVal])

    # Close datasets
    Band = None
    Output = None

def save_greece_boundary_tiffs():
    x = gpd.read_file("data/geo_data/gadm36_GRC_shp/gadm36_GRC_1.shp").to_crs("EPSG:4326")
    x.boundary.to_file("data/geo_data/gadm36_GRC_shp/gadm36_GRC_1_boundary.shp", driver='ESRI Shapefile')


def create_local_tiffs():
    # X = gpd.read_file('greece_quakes.shp').to_crs("EPSG:4326")
    import util.catalogue_processing as cp

    #catalogue = cp.combine_catalogues(_rebuild=False)
    #catalogue["DATE"] = pd.to_datetime(catalogue["DATE"])
    #catalogue = gpd.GeoDataFrame( catalogue, geometry=gpd.points_from_xy(catalogue.LON, catalogue.LAT), crs="EPSG:4326" )
    #catalogue[ catalogue["_I_buffer"] == 1]
    # create bands
    #bins = [2, 4, 5.5, np.inf]
    #labels = ["low", "medium", "high"]
    #catalogue["M_group"] = pd.cut(catalogue['MAG'], bins=bins, labels=labels)
    #catalogue.groupby("M_group").count()

    #for g in catalogue.M_group.unique():
    #    catalogue[catalogue.M_group == g][["MAG", "geometry"]].to_file(f'greece_quakes_{g}.shp', driver='ESRI Shapefile')

    olympia = pd.DataFrame({'longitude': [config.OLYMPIA_GEO.LON], 'latitude': [config.OLYMPIA_GEO.LAT]})
    olympia = gpd.GeoDataFrame(olympia,
                               geometry=gpd.points_from_xy(olympia.longitude, olympia.latitude, crs="EPSG:4326"))
    olympia.geometry = olympia.to_crs("EPSG:32634").geometry.buffer(1 * 1000).to_crs("EPSG:4326")
    olympia.to_file('greece_olympia.shp', driver='ESRI Shapefile')

    # also save buffer
    olympia.geometry = olympia.to_crs("EPSG:32634").geometry.buffer(100 * 1000).to_crs("EPSG:4326")
    olympia.boundary.to_file('greece_olympia_buffer.shp', driver='ESRI Shapefile')

    #save_greece_boundary_tiffs()
    #rasterize_shapefile(InputVector="data/geo_data/gadm36_GRC_shp/gadm36_GRC_1_boundary.shp", OutputImage=f'greece_subareas_1.tif', RefImage='greece_soil.tif')
    # rasterize_shapefile(InputVector="data/geo_data/gadm36_GRC_shp/gadm36_GRC_1_boundary.shp", OutputImage=f'greece_subareas_1.tif', RefImage='greece_soil.tif')
    # rasterize_shapefile(InputVector="data/geo_data/gadm36_GRC_shp/gadm36_GRC_2.shp", OutputImage=f'greece_subareas_2.tif', RefImage='greece_soil.tif')

    rasterize_shapefile(InputVector=f'greece_olympia.shp', OutputImage=f'greece_olympia.tif', RefImage='greece_soil.tif')
    rasterize_shapefile(InputVector=f'greece_olympia_buffer.shp', OutputImage=f'greece_olympia_buffer.tif', RefImage='greece_soil.tif')
    #for g in ["low", "medium", "high"]:
    #    rasterize_shapefile(InputVector=f'greece_quakes_{g}.shp', OutputImage=f'greece_quakes_{g}.tif', RefImage='greece_soil.tif')
    rasterize_shapefile(InputVector='data/geo_data/faults/Fault_projection.shp', OutputImage='greece_faults.tif', RefImage='greece_soil.tif')

    rasterize_shapefile(InputVector='data/geo_data/faults/Faults_line.shp', OutputImage='greece_fault_lines.tif', RefImage='greece_soil.tif')
    # rasterize_shapefile(InputVector='data/geo_data/TectonicPlateBoundaries/TectonicPlateBoundaries.shp', OutputImage='greece_plates.tif', RefImage='greece_soil.tif')


def get_colors(lower_color, upper_color, num_colors, color_map):
    cmap = plt.get_cmap(color_map)
    colors = cmap(np.linspace(lower_color, upper_color, num_colors))
    hexes = [matplotlib.colors.rgb2hex(x) for x in colors]

    return hexes

def plot_topologies():
    elevation = ee.Image('USGS/SRTMGL1_003')
    soil = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02")
    quakes_low = ee.Image("projects/seismic-risk-coursework1/assets/greece_quakes_low").rename(['quakes_low'])
    quakes_med = ee.Image("projects/seismic-risk-coursework1/assets/greece_quakes_medium").rename(['quakes_med'])
    quakes_high = ee.Image("projects/seismic-risk-coursework1/assets/greece_quakes_high").rename(['quakes_high'])

    faults = ee.Image("projects/seismic-risk-coursework1/assets/greece_faults").rename(['faults'])
    plates = ee.Image("projects/seismic-risk-coursework1/assets/greece_plates").rename(['plates'])
    olympia = ee.Image("projects/seismic-risk-coursework1/assets/greece_olympia").rename(['olympia'])
    buffer = ee.Image("projects/seismic-risk-coursework1/assets/greece_olympia_buffer").rename(['buffer'])

    country = ee.Image("projects/seismic-risk-coursework1/assets/greece_country").rename(['country'])
    county = ee.Image("projects/seismic-risk-coursework1/assets/greece_subareas_1").rename(['county'])
    blocks = ee.Image("projects/seismic-risk-coursework1/assets/greece_subareas_2").rename(['blocks'])

    elevation_params = {'min': 0, 'max': 3000, 'palette': get_colors(0.2, 1, 8, "OrRd"), 'opacity': 0.7}
    soil_params = {"bands": ["b30"], "min": 1.0, "max": 12.0, 'palette': get_colors(0, 1, 12, 'PuRd'), 'opacity': 0.5}
    earthquake_colors = get_colors(0.5, 0.95, 3, 'Reds')
    print("elevation", get_colors(0.2, 1, 8, "OrRd"))
    print("earthquake_colors", earthquake_colors)
    print("soil", get_colors(0, 1, 12, 'PuRd'))

    print("plates", get_colors(0.99, 1, 2, "OrRd")[0])
    faults_params = {"bands": ["faults"], 'min': 6, 'max': 8, 'palette': [earthquake_colors[0]], 'opacity': 0.5}
    quakes_low_params = {"bands": ["quakes_low"], 'palette': [earthquake_colors[0]], 'opacity': 1}
    quakes_med_params = {"bands": ["quakes_med"], 'palette': [earthquake_colors[1]], 'opacity': 1}
    quakes_high_params = {"bands": ["quakes_high"], 'palette': [earthquake_colors[2]], 'opacity': 1}
    plates_params = {"bands": ["plates"], 'palette': [get_colors(0.99, 1, 2, "OrRd")[0]], 'opacity': 1}
    olympia_params = {"bands": ["olympia"], 'palette': ['000000'], 'opacity': 1}
    buffer_params = {"bands": ["buffer"], 'palette': ['000000'], 'opacity': 1}
    county_params = {"bands": ["county"], 'palette': ['000000'], 'opacity': 1}

    elevation = elevation.getMapId(elevation_params)["image"]
    soil = soil.getMapId(soil_params)["image"]
    faults = faults.getMapId(faults_params)["image"]
    plates = plates.getMapId(plates_params)["image"]
    olympia = olympia.getMapId(olympia_params)["image"]
    buffer = buffer.getMapId(buffer_params)["image"]
    county = county.getMapId(county_params)["image"]
    #quakes_low = quakes_low.getMapId(quakes_low_params)["image"]
    quakes_med = quakes_med.getMapId(quakes_med_params)["image"]
    quakes_high = quakes_high.getMapId(quakes_high_params)["image"]

    crs = "EPSG:4326"
    region = ee.Geometry.BBox(config.BUFFER_RANGE.MIN_LON,
                              config.BUFFER_RANGE.MIN_LAT,
                              config.BUFFER_RANGE.MAX_LON,
                              config.BUFFER_RANGE.MAX_LAT)
    m = geemap.Map(center=[config.OLYMPIA_GEO.LON, config.OLYMPIA_GEO.LAT], zoom=8)
    # maxValue = img.reduce(ee.Reducer.max())

    elevation = elevation.updateMask(elevation.neq(0))
    soil = soil.updateMask(soil.neq(0))
    faults = faults.updateMask(faults.neq(0))
    olympia = olympia.updateMask(olympia.neq(0))
    buffer = buffer.updateMask(buffer.neq(0))
    plates = plates.updateMask(plates.neq(0))
    county = county.updateMask(county.neq(0))
    #quakes_low = quakes_low.updateMask(quakes_low.neq(0))
    quakes_med = quakes_med.updateMask(quakes_med.neq(0))
    quakes_high = quakes_high.updateMask(quakes_high.neq(0))

    m.addLayer(elevation, elevation_params, name="Elevation")
    m.addLayer(soil, soil_params, name="Soil")
    #m.addLayer(quakes_low, quakes_low_params, name="Quakes M3-4")
    m.addLayer(quakes_med, quakes_med_params, name="Quakes M4-5.5")
    m.addLayer(quakes_high, quakes_high_params, name="Quakes M5.5+")
    m.addLayer(faults, faults_params, name="Faults")
    m.addLayer(plates, plates_params, name="Tectonic Boundaries")
    m.addLayer(olympia, olympia_params, name="Olympia")
    m.addLayer(buffer, buffer_params, name="Olympia")
    m.addLayer(county, county_params, name="Greece")

    m.to_html(filename="output/zones_disagg.html", title="My Map", width="100%", height="880px")

    # m.layer_to_image("Zones", output="zones.jpg", region=region,scale=10)

if __name__ == "__main__":
    ee.Authenticate()
    ee.Initialize(project="seismic-risk-coursework1")


    #create_local_tiffs()
    plot_topologies()

