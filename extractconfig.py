__author__     = "Vijay Raj Franklin"
__license__    = "Apache License 2.0"
__version__    = "1.0.0"
__maintainer__ = "Vijay Raj Franklin"
__email__      = "franklynece@gmail.com"

import logging
logger = logging.getLogger(__name__)

import json

class ConfigParser:
    '''
        Extract the configuration from the config file and 
        stores it in an internal dictionary object.
    '''
    def __init__(self,cfgjson):
        self.cfgjson = cfgjson

    def extract(self):
        with open(self.cfgjson) as f:
            cfg = json.load(f)
            logger.info("[SUCCESS] Extracted the configuration from the config file : %s", self.cfgjson)
            logger.info(json.dumps(cfg, indent=4, sort_keys=False))
            # ###
            return cfg
