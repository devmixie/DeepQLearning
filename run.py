__author__     = "Vijay Raj Franklin"
__license__    = "Apache License 2.0"
__version__    = "1.0.0"
__maintainer__ = "Vijay Raj Franklin"
__email__      = "franklynece@gmail.com"

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

import os
import signal
import argparse
import shutil

#let's import our main game class.
from game import Game

def main(cfgjson,modeldir,modelfl):
    def setup_model_dir():
        if os.path.isdir(modeldir):
            shutil.rmtree(modeldir,ignore_errors=False)
        os.makedirs(modeldir, exist_ok=True)
    # ###
    if not modelfl:
        setup_model_dir()
    Game(cfgjson,modeldir,modelfl).run() #start the game training with our config file.

if __name__ == "__main__":
    def gracefulexit(signum, frame):
        logger.warning ("Ctrl-c was pressed. Program terminated gracefully.")
        exit(1)
    # ##
    signal.signal(signal.SIGINT, gracefulexit) # register callback to ctrl+c, so we can handle this gracefully and 
    
    #setting up argparse to pass command line arguments to the program.
    parser = argparse.ArgumentParser(description='Program that automatically learns to play games.')
    parser.add_argument('-c','--cfgjson',
                         required = False,
                         default  = './cfg.json',
                         help='JSON file containing the configuration',
                       )
    parser.add_argument('-d','--modeldir',
                         required = False,
                         default  = './model',
                         help='Directory to save the trained models',
                       )
    parser.add_argument('-m','--modelfl',
                         required = False,
                         default  = None,
                         help='Trained model to load',
                       )
    args  = vars(parser.parse_args())
    logger.info("started")
    main(args['cfgjson'],args['modeldir'],args['modelfl'])
        