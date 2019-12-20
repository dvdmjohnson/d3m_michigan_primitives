#!/usr/bin/env python3

#convenience script for generating the entire primitive JSON file structure for spider

import subprocess
import os, os.path
import shutil
import sys
import pkg_resources
import json
from spider.pipelines import *

#clear out any existing directory
if os.path.isdir("Michigan"):
    shutil.rmtree("Michigan")

os.makedirs("Michigan")

version = pkg_resources.get_distribution("spider").version

primitives = {
    'd3m.primitives.regression.owl_regression.Umich' : [OWLRegressionPipeline, OWLRegressionPipelineChallenge1, OWLRegressionPipelineChallenge2, OWLRegressionPipelineChallenge3],
    'd3m.primitives.learner.goturn.Umich' : None,
    'd3m.primitives.feature_extraction.vgg16.Umich' : [FeaturizationVGG16Pipeline],
    'd3m.primitives.feature_extraction.i3d.Umich' : None,
    'd3m.primitives.feature_extraction.audio_featurization.Umich' : [FeaturizationAudioFeaturizationPipeline],
    'd3m.primitives.feature_extraction.log_mel_spectrogram.Umich' : None,
    'd3m.primitives.data_preprocessing.audio_slicer.Umich' : None,
    'd3m.primitives.similarity_modeling.rfd.Umich' : None,
    'd3m.primitives.data_compression.go_dec.Umich' : [GO_DECPipeline],
    'd3m.primitives.data_compression.pcp_ialm.Umich' : [PCP_IALMPipeline],
    'd3m.primitives.data_compression.rpca_lbd.Umich' : [RPCA_LBDPipeline],
    'd3m.primitives.data_compression.grasta.Umich' : [GRASTAPipeline, GRASTAPipelineChallenge1, GRASTAPipelineChallenge2],
    'd3m.primitives.data_compression.grasta_masked.Umich' : None,
    'd3m.primitives.data_compression.grouse.Umich' : None,
    'd3m.primitives.clustering.kss.Umich' : [KSSPipeline],
    'd3m.primitives.clustering.ekss.Umich' : [EKSSPipeline],
    'd3m.primitives.clustering.ssc_admm.Umich' : [SSCADMMPipeline],
    'd3m.primitives.clustering.ssc_cvx.Umich' : [SSCCVXPipeline],
    'd3m.primitives.clustering.ssc_omp.Umich' : [SSCOMPPipeline],
    'd3m.primitives.data_preprocessing.trecs.Umich' : None}

for prim in primitives.keys():
    path = os.path.join("Michigan", prim)
    os.makedirs(path)
    path = os.path.join(path, version)
    os.makedirs(path)

    com = "python3 -m d3m index describe -i 4 " + prim + " > " + os.path.join(path, "primitive.json")
    print('Running command: %s' % str(com))
    subprocess.check_call(com, shell=True)

    #now make pipelines
    if primitives[prim] is not None:
        plpath = os.path.join(path, 'pipelines')
        os.makedirs(plpath)
        
        pls = primitives[prim]
        for pl in pls:
            instance = pl()
            json_info = instance.get_json()
            instanceid = instance.get_id()

            instancepath = os.path.join(plpath, instanceid)
            with open(instancepath + ".json", 'w') as file:
                file.write(json_info)
                file.close()
            
            # meta = instance.meta_info
            # with open(instancepath + ".meta", 'w') as file:
            #     json.dump(meta, file, indent = 4)
            #     file.close()
    
    

