#!/usr/bin/env python3

"""
Convenience script for generating the entire primitive JSON file structure for spider.
"""

import subprocess
import os
import shutil
import sys
import pkg_resources
import json
from spider.pipelines import *

# Ensure that this script operates in the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
os.chdir(PROJ_DIR)

# Clear out any existing directory
if os.path.isdir("Michigan"):
    shutil.rmtree("Michigan")

os.makedirs("Michigan")

# Get Spider package version (read from setup.py)
version = pkg_resources.get_distribution('spider').version
# Get fully-qualified module names for each Spider primitive (read from setup.py) 
# E.g. d3m.primitives.learner.goturn.Umich
primitive_module_names = ['d3m.primitives.{}'.format(x) for x in pkg_resources.get_entry_map('spider')['d3m.primitives'].keys()]

# Stores information required to generate the command to run each pipeline
pipeline_cmd_template_info = {
    OWLRegressionPipeline: {
        'primitive': 'd3m.primitives.regression.owl_regression.Umich',
        'command': "python3 -m d3m runtime -v /volumes fit-produce -p Michigan/{primitive}/{version}/pipelines/{instanceid}.json -r /datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json -i /datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json -o /dev/null -O Michigan/{primitive}/0.0.5/pipeline_runs/{pipeline}.yaml"
    },
    # OWLRegressionPipelineChallenge1: {
    #     'primitive': 'd3m.primitives.regression.owl_regression.Umich',
    #     'command': "python3 -m d3m runtime -v /volumes fit-produce -p Michigan/{primitive}/{version}/pipelines/{instanceid}.json -r /datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json -i /datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json -o /dev/null -O Michigan/{primitive}/0.0.5/pipeline_runs/{pipeline}.yaml"
    # },
    OWLRegressionPipelineChallenge2: {
        'primitive': 'd3m.primitives.regression.owl_regression.Umich',
        'command': "python3 -m d3m runtime -v /volumes fit-produce -p Michigan/{primitive}/{version}/pipelines/{instanceid}.json -r /datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json -i /datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json -o /dev/null -O Michigan/{primitive}/0.0.5/pipeline_runs/{pipeline}.yaml"
    },
    OWLRegressionPipelineChallenge3: {
        'primitive': 'd3m.primitives.regression.owl_regression.Umich',
        'command': "python3 -m d3m runtime -v /volumes fit-produce -p Michigan/{primitive}/{version}/pipelines/{instanceid}.json -r /datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json -i /datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json -o /dev/null -O Michigan/{primitive}/0.0.5/pipeline_runs/{pipeline}.yaml"
    },
    # FeaturizationVGG16Pipeline: {
    #     'primitive': 'd3m.primitives.feature_extraction.vgg16.Umich',
    #     'command': "python3 -m d3m runtime -v /volumes fit-produce -p Michigan/{primitive}/{version}/pipelines/{instanceid}.json -r /datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json -i /datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json -o /dev/null -O Michigan/{primitive}/0.0.5/pipeline_runs/{pipeline}.yaml"
    # },
    # GO_DECPipeline: {
    #     'primitive': 'd3m.primitives.data_compression.go_dec.Umich',
    #     'command': "python3 -m d3m runtime -v /volumes fit-produce -p Michigan/{primitive}/{version}/pipelines/{instanceid}.json -r /datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json -i /datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json -o /dev/null -O Michigan/{primitive}/0.0.5/pipeline_runs/{pipeline}.yaml"
    # },
    # PCP_IALMPipeline: {
    #     'primitive': 'd3m.primitives.data_compression.pcp_ialm.Umich',
    #     'command': "python3 -m d3m runtime -v /volumes fit-produce -p Michigan/{primitive}/{version}/pipelines/{instanceid}.json -r /datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json -i /datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json -o /dev/null -O Michigan/{primitive}/0.0.5/pipeline_runs/{pipeline}.yaml"
    # },
    # RPCA_LBDPipeline: {
    #     'primitive': 'd3m.primitives.data_compression.rpca_lbd.Umich',
    #     'command': "python3 -m d3m runtime -v /volumes fit-produce -p Michigan/{primitive}/{version}/pipelines/{instanceid}.json -r /datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json -i /datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json -o /dev/null -O Michigan/{primitive}/0.0.5/pipeline_runs/{pipeline}.yaml"
    # },
    GRASTAPipeline: {
        'primitive': 'd3m.primitives.data_compression.grasta.Umich',
        'command': "python3 -m d3m runtime -v /volumes fit-produce -p Michigan/{primitive}/{version}/pipelines/{instanceid}.json -r /datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json -i /datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json -o /dev/null -O Michigan/{primitive}/0.0.5/pipeline_runs/{pipeline}.yaml"
    },
    GRASTAPipelineChallenge1: {
        'primitive': 'd3m.primitives.data_compression.grasta.Umich',
        'command': "python3 -m d3m runtime -v /volumes fit-produce -p Michigan/{primitive}/{version}/pipelines/{instanceid}.json -r /datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json -i /datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json -o /dev/null -O Michigan/{primitive}/0.0.5/pipeline_runs/{pipeline}.yaml"
    },
    # GRASTAPipelineChallenge2: {
    #     'primitive': 'd3m.primitives.data_compression.grasta.Umich',
    #     'command': "python3 -m d3m runtime -v /volumes fit-produce -p Michigan/{primitive}/{version}/pipelines/{instanceid}.json -r /datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json -i /datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json -o /dev/null -O Michigan/{primitive}/0.0.5/pipeline_runs/{pipeline}.yaml"
    # },
    KSSPipeline: {
        'primitive': 'd3m.primitives.clustering.kss.Umich',
        'command': "python3 -m d3m runtime -v /volumes fit-produce -p Michigan/{primitive}/{version}/pipelines/{instanceid}.json -r /datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json -i /datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json -o /dev/null -O Michigan/{primitive}/0.0.5/pipeline_runs/{pipeline}.yaml"
    },
    EKSSPipeline: {
        'primitive': 'd3m.primitives.clustering.ekss.Umich',
        'command': "python3 -m d3m runtime -v /volumes fit-produce -p Michigan/{primitive}/{version}/pipelines/{instanceid}.json -r /datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json -i /datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json -o /dev/null -O Michigan/{primitive}/0.0.5/pipeline_runs/{pipeline}.yaml"
    },
    SSCADMMPipeline: {
        'primitive': 'd3m.primitives.clustering.ssc_admm.Umich',
        'command': "python3 -m d3m runtime -v /volumes fit-produce -p Michigan/{primitive}/{version}/pipelines/{instanceid}.json -r /datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json -i /datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json -o /dev/null -O Michigan/{primitive}/0.0.5/pipeline_runs/{pipeline}.yaml"
    },
    SSCCVXPipeline: {
        'primitive': 'd3m.primitives.clustering.ssc_cvx.Umich',
        'command': "python3 -m d3m runtime -v /volumes fit-produce -p Michigan/{primitive}/{version}/pipelines/{instanceid}.json -r /datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json -i /datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json -o /dev/null -O Michigan/{primitive}/0.0.5/pipeline_runs/{pipeline}.yaml"
    },
    SSCOMPPipeline: {
        'primitive': 'd3m.primitives.clustering.ssc_omp.Umich',
        'command': "python3 -m d3m runtime -v /volumes fit-produce -p Michigan/{primitive}/{version}/pipelines/{instanceid}.json -r /datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json -i /datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json -o /dev/null -O Michigan/{primitive}/0.0.5/pipeline_runs/{pipeline}.yaml"
    },
}

pipeline_cmds = []

for prim in primitive_module_names:
    prim_path = os.path.join("Michigan", prim)
    os.makedirs(prim_path)
    version_path = os.path.join(prim_path, version)
    os.makedirs(version_path)
    # Make pipeline run directory
    pipeline_runs_path = os.path.join(version_path, 'pipeline_runs')
    os.makedirs(pipeline_runs_path)

    com = "python3 -m d3m index describe -i 4 " + prim + " > " + os.path.join(version_path, "primitive.json")
    print('Running command: %s' % str(com))
    subprocess.check_call(com, shell=True)

# Now make pipelines
for pl in pipeline_cmd_template_info.keys():
    pl_name = pl.__name__
    instance = pl()
    json_info = instance.get_json()
    instanceid = instance.get_id()
    dataset = instance.dataset
    prim = pipeline_cmd_template_info[pl]['primitive']

    print(prim, pl_name, '->', instanceid)

    # Fill out the template for this pipeline's run command
    pipeline_cmd = pipeline_cmd_template_info[pl]['command'].format(
        version=version,
        primitive=prim,
        pipeline=pl_name,
        instanceid=instanceid,
        dataset=dataset
    )
    pipeline_cmds.append(pipeline_cmd)

    prim_path = os.path.join("Michigan", prim)
    version_path = os.path.join(prim_path, version)
    plpath = os.path.join(version_path, 'pipelines')
    if not os.path.isdir(plpath):
        os.makedirs(plpath)
    instancepath = os.path.join(plpath, instanceid)
    with open(instancepath + ".json", 'w') as file:
        file.write(json_info)

# Save the run pipeline commands to a file
with open('run_pipeline_cmds.txt', 'w') as f:
    f.write('\n'.join(pipeline_cmds))
    f.write('\n')
