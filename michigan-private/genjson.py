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

fit_score_command = 'python3 -m d3m runtime -v /volumes fit-score -p Michigan/{primitive}/{version}/pipelines/{instanceid}.json -r /datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json -i /datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json -a /datasets/seed_datasets_current/{dataset}/SCORE/dataset_TEST/datasetDoc.json -o /dev/null -O Michigan/{primitive}/0.0.5/pipeline_runs/{pipeline}.yaml > pipeline_results/{pipeline}.txt'

# Stores information required to generate the command to run each pipeline
pipeline_cmd_template_info = {
    OWLRegressionPipeline: 'd3m.primitives.regression.owl_regression.Umich',
    # OWLRegressionPipelineChallenge1: 'd3m.primitives.regression.owl_regression.Umich',
    OWLRegressionPipelineChallenge2: 'd3m.primitives.regression.owl_regression.Umich',
    OWLRegressionPipelineChallenge3: 'd3m.primitives.regression.owl_regression.Umich',
    # FeaturizationVGG16Pipeline: 'd3m.primitives.feature_extraction.vgg16.Umich',
    # GO_DECPipeline: 'd3m.primitives.data_compression.go_dec.Umich',
    # PCP_IALMPipeline: 'd3m.primitives.data_compression.pcp_ialm.Umich',
    # RPCA_LBDPipeline: 'd3m.primitives.data_compression.rpca_lbd.Umich',
    # GROUSEPipeline: 'd3m.primitives.data_compression.grouse.Umich',
    GRASTAPipeline: 'd3m.primitives.data_compression.grasta.Umich',
    GRASTAPipelineChallenge1: 'd3m.primitives.data_compression.grasta.Umich',
    # GRASTAPipelineChallenge2: 'd3m.primitives.data_compression.grasta.Umich',
    KSSPipeline: 'd3m.primitives.clustering.kss.Umich',
    EKSSPipeline: 'd3m.primitives.clustering.ekss.Umich',
    SSCADMMPipeline: 'd3m.primitives.clustering.ssc_admm.Umich',
    SSCCVXPipeline: 'd3m.primitives.clustering.ssc_cvx.Umich',
    SSCOMPPipeline: 'd3m.primitives.clustering.ssc_omp.Umich',
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
    prim = pipeline_cmd_template_info[pl]

    print(prim, pl_name, '->', instanceid)

    # Fill out the template for this pipeline's run command
    pipeline_cmd = fit_score_command.format(
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

# Generate folder to store pipeline results
if not os.path.isdir('pipeline_results'):
	os.makedirs('pipeline_results')
