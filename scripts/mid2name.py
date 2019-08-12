import os
import sys
import json
import pickle

import click
import logging

import numpy as np
import urllib.parse
import urllib.request

from tqdm import tqdm

from SPARQLWrapper import SPARQLWrapper, JSON

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GOOGLE_API_KEY = 'GOOGLE_API_KEY'

OUTPUT_HELP_MSG = "The output path for the output files"
FILES_HELP_MSG = "The path to the mappings files"
APIKEY_HELP_MSG = """An api key for the google api. Alternatively
                  the {GOOGLE_API_KEY} environment variable must be set."""


def get_results(endpoint_url, query):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def create_mid2name_dict(files, output, apikey):
    e2id = dict(np.load(files), allow_pickle=True)
    del e2id["allow_pickle"]
    mid2name = dict()

    sys.setrecursionlimit(len(e2id)+1000)

    api_key = apikey or os.environ.get(GOOGLE_API_KEY)

    if not api_key:
        raise Exception("""
                        You have to pass the --apikey option or set the
                        {GOOGLE_API_KEY} environment variable.
                        """)

    service_url = 'https://kgsearch.googleapis.com/v1/entities:search'

    for Mid in tqdm(e2id):
        mid_num = Mid[3:]

        # try  with wikidata first
        # query to get corresponding entity name
        endpoint_url = "https://query.wikidata.org/sparql"
        query = """PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                PREFIX wikibase: <http://wikiba.se/ontology#>

                SELECT  ?sLabel WHERE {
                ?s wdt:P646 "/m/"""\
                + mid_num +\
                """" .

                  SERVICE wikibase:label {
                  bd:serviceParam wikibase:language "en" .
                  }
                }
                LIMIT 1
                """

        results = get_results(endpoint_url, query)
        if (len(results["results"]["bindings"]) == 0):
            params = {
                'ids': Mid,
                'limit': 10,
                'indent': True,
                'key': api_key,
            }
            url = service_url + '?' + urllib.parse.urlencode(params)
            response = json.loads(urllib.request.urlopen(url).read())
            for element in response['itemListElement']:
                # Add entity's name if possible
                if 'name' in element['result']:
                    # Add queried value with preprocessed id as key
                    mid2name[Mid] = element['result']['name']
                # Else description
                elif 'detailedDescription' in element['result']:
                    mid2name[Mid] = (element['result']
                                     ['detailedDescription']
                                     ['articleBody'])
                else:
                    logger.warning("Couldn't find mapping for %s -> %s", str(Mid), str(e2id[Mid])) # noqa
                    logger.warning("Exiting now...")
                    return

        else:
            # Add queried value with preprocessed id as key
            mid2name[Mid] = (results["results"]
                             ["bindings"][0]["sLabel"]["value"])

    # Check if output directory exists otherwise create.
    if not os.path.isdir(output):
        logger.warning("Output directory does not exist: %s", output) # noqa
        logger.warning("Creating path...")
        os.makedirs(output)

    filename = 'mid2name.pkl'
    output_file = os.path.join(output, filename)

    logger.infos('Writing output file...')
    with open(output_file, 'wb') as f:
        pickle.dump(mid2name, f, pickle.HIGHEST_PROTOCOL)

    logger.info('Sanity check: Loading dictionaries.')
    with open(output_file, 'rb') as f:
        loaded_data = pickle.load(f)
        logger.info(loaded_data)


@click.command()
@click.option('--files', help=FILES_HELP_MSG)
@click.option('--out', help=OUTPUT_HELP_MSG)
@click.option('--apikey', default=None, help=APIKEY_HELP_MSG)
def main(files, out, apikey):
    """Simple script that creates a dictionary mapping freebase mIDs to text"""
    create_mid2name_dict(files, out, apikey)

#pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
