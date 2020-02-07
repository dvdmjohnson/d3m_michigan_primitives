#!/usr/bin/env python3

"""
Convenience script for generating the entire primitive JSON file structure for spider.
"""

import json
import os
import pkg_resources
import shutil
from subprocess import check_output, CalledProcessError

import spider.pipelines as spls

from utils import BashCommandWorkerPool


def main():
    # Ensure that this script operates in the project root
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJ_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    os.chdir(PROJ_DIR)

    # Get remote and current commit info from the dataset repo
    DATASETS_DIR = '/datasets'
    commit_hash, origin_url = get_git_info(DATASETS_DIR)
    # Make sure the current dataset repo matches the version supported by this project
    with open(os.path.join(PROJ_DIR, 'michigan-private', 'dataset_repo_info.json')) as f:
        dataset_repo_info = json.load(f)
    assert commit_hash == dataset_repo_info['commit_hash'], \
        'Expected commit hash {} at {}, but got {}'.format(dataset_repo_info['commit_hash'], DATASETS_DIR, commit_hash)
    assert origin_url == dataset_repo_info['origin_url'], \
        'Expected origin URL {} at {}, but got {}'.format(dataset_repo_info['origin_url'], DATASETS_DIR, origin_url)

    # Clear out any existing directory
    if os.path.isdir('Michigan'):
        shutil.rmtree('Michigan')

    os.makedirs('Michigan')

    # Get Spider package version (read from setup.py)
    version = pkg_resources.get_distribution('spider').version
    # Get fully-qualified module names for each Spider primitive (read from setup.py)
    # E.g. d3m.primitives.learner.goturn.Umich
    primitive_module_names = ['d3m.primitives.{}'.format(x) for x in
                              pkg_resources.get_entry_map('spider')['d3m.primitives'].keys()]

    # List of pipelines to run and export
    pipelines_to_run = [
        spls.EKSSOneHundredPlantsMarginPipeline,
        # spls.GODECHandgeometryPipeline,
        spls.GRASTAAutoMPGPipeline,
        spls.GRASTAAutoPricePipeline,
        # spls.GROUSEAutoMPGPipeline,
        # spls.I3DHMDBActionRecognitionPipeline,
        spls.KSSOneHundredPlantsMarginPipeline,
        spls.OWLRegressionAutoPricePipeline,
        # spls.PCPIALMHandgeometryPipeline,
        # spls.RPCALBDHandgeometryPipeline,
        # spls.SSCADMMOneHundredPlantsMarginPipeline,
        # spls.SSCCVXOneHundredPlantsMarginPipeline,
        # spls.SSCOMPOneHundredPlantsMarginPipeline,
        # spls.VGG16HandgeometryPipeline,
    ]

    primitive_cmds = []
    pipeline_cmds = []

    for prim in primitive_module_names:
        prim_path = os.path.join('Michigan', prim)
        os.makedirs(prim_path)
        version_path = os.path.join(prim_path, version)
        os.makedirs(version_path)
        # Make pipeline run directory
        pipeline_runs_path = os.path.join(version_path, 'pipeline_runs')
        os.makedirs(pipeline_runs_path)

        cmd = 'python3 -m d3m index describe -i 4 ' + prim + ' > ' + os.path.join(version_path, 'primitive.json')
        primitive_cmds.append(cmd)

    # Run describe commands in parallel
    pool = BashCommandWorkerPool(8)
    for cmd in primitive_cmds:
        pool.add_work(cmd)
    pool.join()

    # Now make pipelines
    for pipeline in pipelines_to_run:
        pl_name = pipeline.__name__
        json_info = pipeline.get_json()
        instanceid = pipeline.get_id()
        dataset = pipeline.get_dataset_class().get_dataset_name()
        prim = pipeline.get_primitive_entry_point()

        print(prim, pl_name, '->', instanceid)

        # Fill out the template for this pipeline's run command
        template = pipeline.get_dataset_class().get_fit_score_command_template()
        pipeline_cmd = template.format(
            version=version,
            primitive=prim,
            pipeline=pl_name,
            instanceid=instanceid,
            dataset=dataset
        )
        pipeline_cmds.append(pipeline_cmd)

        prim_path = os.path.join('Michigan', prim)
        version_path = os.path.join(prim_path, version)
        plpath = os.path.join(version_path, 'pipelines')
        if not os.path.isdir(plpath):
            os.makedirs(plpath)
        instancepath = os.path.join(plpath, instanceid)
        with open(instancepath + '.json', 'w') as file:
            file.write(json_info)

    # Save the run pipeline commands to a file
    with open('run_pipeline_cmds.txt', 'w') as f:
        f.write('\n'.join(pipeline_cmds))
        f.write('\n')

    # Generate folder to store pipeline results
    if not os.path.isdir('pipeline_results'):
        os.makedirs('pipeline_results')


def get_git_info(git_dir):
    """Gets the commit hash and origin URL of the Git repo at the given location.

    :param git_dir: The path to the Git repository
    :return: tuple containing the commit hash and the origin url
    """
    try:
        commit_hash_bytes = check_output('cd {} && git rev-parse HEAD'.format(git_dir), shell=True)
    except CalledProcessError:
        raise RuntimeError('Failed to get commit hash of {}'.format(git_dir))
    commit_hash = commit_hash_bytes.strip().decode()

    try:
        origin_url_bytes = check_output('cd {} && git config --get remote.origin.url'.format(git_dir), shell=True)
    except CalledProcessError:
        raise RuntimeError('Failed to get remote origin URL of {}'.format(git_dir))
    origin_url = origin_url_bytes.strip().decode()

    return commit_hash, origin_url


if __name__ == '__main__':
    main()
