import argparse
import logging
import json

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S') 

parser = argparse.ArgumentParser(description='Neat way of storing the experiment configuration')
parser.add_argument('--exp_config', dest='EXP_CONFIG', action='store', help='full path to JSON that will contain experiment configuration', required=True)
args = parser.parse_args()

logging.info("Storing config to file: {}".format(args.EXP_CONFIG))
exp_config = dict()

exp_config['communities'] = [0, 1]
exp_config['edge_weights'] = [1, 3, 5]
exp_config['node_weights'] = [2, 4, 6]
exp_config['sem_weights'] = [0.3, 0.5, 0.7]
exp_config['other_thresholds'] = [0, 0.05, 0.1, 0.2, 0.4, 1]
exp_config['top_k'] = [3, 5, 7]

with open(args.EXP_CONFIG, 'w') as json_file:
    json.dump(exp_config, json_file)
logging.info("Finished")
