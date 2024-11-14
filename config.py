import util.catalogue_declustering as catalogue_declustering
from dotmap import DotMap

MAG_MIN = 2
MAX_YEAR = 2022
BUFFER_RANGE = DotMap({
    "MIN_LAT": 34.7188863,
    "MAX_LAT": 41.7488889,
    "MIN_LON": 19.1127535,
    "MAX_LON":  29.68381,

})

STUDY_BUFFER = 100

DECLUSTER_CONFIGS = DotMap({
        "time_distance_window": catalogue_declustering.UhrhammerWindow,
        "fs_time_prop": 1/2,
    })

COMPLETENESS_CONFIGS = {
        "magnitude_bin": 1,
        "time_bin": 10,
        "increment_lock": False
    }

#37.6383,21.63
OLYMPIA_GEO = DotMap({
    "LAT": 37.6383,
    "LON": 21.63
})

CRS = "EPSG:4326"
