#!/usr/bin/env python3

"""
Convenience script for generating the entire primitive JSON file structure for spider.
"""

import os
import pkg_resources
import shutil
import subprocess

from spider.pipelines import *

# Ensure that this script operates in the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
os.chdir(PROJ_DIR)

# Clear out any existing directory
if os.path.isdir('Michigan'):
    shutil.rmtree('Michigan')

os.makedirs('Michigan')

# Get Spider package version (read from setup.py)
version = pkg_resources.get_distribution('spider').version
# Get fully-qualified module names for each Spider primitive (read from setup.py) 
# E.g. d3m.primitives.learner.goturn.Umich
primitive_module_names = ['d3m.primitives.{}'.format(x) for x in pkg_resources.get_entry_map('spider')['d3m.primitives'].keys()]

default_fit_score_command_template = 'python3 -m d3m runtime -v /volumes fit-score -p Michigan/{primitive}/{version}/pipelines/{instanceid}.json -r /datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json -i /datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json -a /datasets/seed_datasets_current/{dataset}/SCORE/dataset_TEST/datasetDoc.json -o /dev/null -O Michigan/{primitive}/0.0.5/pipeline_runs/{pipeline}.yaml > pipeline_results/{pipeline}.txt'

# List of pipelines to run and export
pipelines_to_run = [
    # AudioFeaturizationUrbansoundPipeline,
    # EKSSOneHundredPlantsMarginPipeline,
    # GODECHandgeometryPipeline,
    GRASTAAutoMPGPipeline,
    GRASTAAutoPricePipeline,
    # GROUSEAutoMPGPipeline,
    # I3DHMDBActioRecognitionPipeline,
    # KSSOneHundredPlantsMarginPipeline,
    OWLRegressionAutoMPGPipeline,
    OWLRegressionAutoPricePipeline,
    OWLRegressionCPS85WagesPipeline,
    # OWLRegressionRadonSeedPipeline,
    # PCPIALMHandgeometryPipeline,
    # RPCALBDHandgeometryPipeline,
    # SSCADMMOneHundredPlantsMarginPipeline,
    # SSCCVXOneHundredPlantsMarginPipeline,
    # SSCOMPOneHundredPlantsMarginPipeline,
    # VGG16HandgeometryPipeline,
]

pipeline_cmds = []

for prim in primitive_module_names:
    prim_path = os.path.join('Michigan', prim)
    os.makedirs(prim_path)
    version_path = os.path.join(prim_path, version)
    os.makedirs(version_path)
    # Make pipeline run directory
    pipeline_runs_path = os.path.join(version_path, 'pipeline_runs')
    os.makedirs(pipeline_runs_path)

    com = 'python3 -m d3m index describe -i 4 ' + prim + ' > ' + os.path.join(version_path, 'primitive.json')
    print('Running command: %s' % str(com))
    subprocess.check_call(com, shell=True)

# Now make pipelines
for pl in pipelines_to_run:
    pl_name = pl.__name__
    instance = pl()
    json_info = instance.get_json()
    instanceid = instance.get_id()
    dataset = instance.dataset
    prim = instance.get_primitive_entry_point()

    print(prim, pl_name, '->', instanceid)

    # Fill out the template for this pipeline's run command
    template = instance.get_fit_score_command_template()
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
