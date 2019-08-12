import click
import logging
from os import path
import os
import pickle
import numpy as np
from dotenv import find_dotenv, load_dotenv

FILE_DIR = path.dirname(__file__)


@click.command()
@click.argument("input_filepath", type=click.Path())
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str = "./data",
         output_filepath: str = "./data") -> None:
    """ Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).

    Parameters:
        input_filepath: relative path to the directory with raw data
        files
        out_file: relative path to the directory for resultsnte
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    raw_data_dir = path.abspath(input_filepath)
    if path.isdir(raw_data_dir):

        processed_data_dir = path.abspath(output_filepath)

        logger.info("start")
        filenames = ["train.txt", "valid.txt", "test.txt"]
        create_index(filenames, raw_data_dir, processed_data_dir)
        prepare_datasets(filenames, raw_data_dir, processed_data_dir)

    else:
        logger.info("File or directory does not exist")

    logger.info("finished")


def create_index(filenames: list, raw_data_dir: str,
                 processed_data_dir: str) -> None:
    """
    maps all entities and relations to a unique id, writes results into
    pickle files

    Parameters:
        filenames: names of all raw data files to be processed
        raw_data_dir: absolute path to the directory with raw data files
        processed_data_dir: absolute path to the directory for results
    """
    entities, relations = set(), set()

    for filename in filenames:
        file_path = path.join(raw_data_dir, filename)
        with open(file_path, 'r') as file:
            for line in file.readlines():
                mid1, relation, mid2 = line.strip().split('\t')
                entities.add(mid1)
                entities.add(mid2)
                relations.add(relation)

    logger = logging.getLogger(__name__)
    logger.info("Found %i different entities", len(entities))
    logger.info("Found %i different relations", len(relations))

    entity_to_id = {entity: i for (i, entity) in enumerate(sorted(entities))}
    relation_to_id = {relation: i for (i, relation) in
                      enumerate(sorted(relations))}

    id_to_entity = {i: entity for entity, i in entity_to_id.items()}
    id_to_relation = {i: relation for relation, i in relation_to_id.items()}

    e2i_path = processed_data_dir + "/entity_to_id.pickle"
    filename_relation_to_id = processed_data_dir + "/relation_to_id.pickle"
    i2e_path = processed_data_dir + "/id_to_entity.pickle"
    filename_id_to_relation = processed_data_dir + "/id_to_relation.pickle"

    os.makedirs(processed_data_dir, exist_ok=True)

    with open(e2i_path, "wb") as handle1:
        pickle.dump(entity_to_id, handle1, protocol=pickle.HIGHEST_PROTOCOL)

    with open(filename_relation_to_id, "wb") as handle2:
        pickle.dump(relation_to_id, handle2, protocol=pickle.HIGHEST_PROTOCOL)

    with open(i2e_path, "wb") as handle3:
        pickle.dump(id_to_entity, handle3, protocol=pickle.HIGHEST_PROTOCOL)

    with open(filename_id_to_relation, "wb") as handle4:
        pickle.dump(id_to_relation, handle4, protocol=pickle.HIGHEST_PROTOCOL)


def prepare_datasets(filenames: list, raw_data_dir: str,
                     processed_data_dir: str) -> None:
    """
    maps the datasets to ids, writes results into pickle files

    Parameters:
        filenames: names of all raw data files to be processed
        raw_data_dir: absolute path to the directory with raw data files
        processed_data_dir: absolute path to the directory for results
    """

    e2i_path = processed_data_dir + "/entity_to_id.pickle"
    filename_relation_to_id = processed_data_dir + "/relation_to_id.pickle"

    with open(e2i_path, "rb") as handle1:
        entity_to_id = pickle.load(handle1)

    with open(filename_relation_to_id, "rb") as handle2:
        relation_to_id = pickle.load(handle2)

    for filename in filenames:
        file_path = path.join(raw_data_dir, filename)
        triples = []

        with open(file_path, 'r') as file:
            for line in file.readlines():
                mid1, relation, mid2 = line.strip().split('\t')
                triples.append([entity_to_id[mid1], relation_to_id[relation],
                               entity_to_id[mid2]])

        filename_new = filename[:-4] + ".pickle"
        filename_output = path.join(processed_data_dir, filename_new)
        with open(filename_output, "wb") as handle:
            pickle.dump(np.array(triples), handle,  # alternative: numpy.save
                        protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automatically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()  # comment out if you want to run Unittests!
    # unittest.main(argv=['first-arg-is-ignored'])  # \n
