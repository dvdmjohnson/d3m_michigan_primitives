# michigan-private

This folder contains scripts specifically for development by the Michigan team. If you're wondering, "private" is just a misnomer from a legacy configuration.

Developers from the Michigan team should use this document as the primary resource for how to run things. Other developers should look at the README in the project root.

## Table of Contents

* [Docker Setup](#docker-setup)
* [Internal Tests and Pipelines](#internal-tests-and-pipelines)
    - [Internal tests](#internal-tests)
    - [Pipelines](#pipelines)
* [Merging Changes](#merging-changes)
* [[POC] Submitting to datadrivendiscovery/primitives](#poc-submitting-to-datadrivendiscoveryprimitives)

## Docker Setup

The current recommended method to install, test and use this library is by installing it within a pregenerated d3m docker image, which already contains up-to-date versions of the full d3m universe, including the various components that our primitives/pipelines depend on.

1. To install Docker, follow the recommended installation procedure for your OS (for Ubuntu, [installing via Docker's repository is recommended](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-using-the-repository)).

2. Build and launch a fresh D3M Docker image:

    ```
    cd d3m_michigan_primitives/michigan-private
    ./docker_build.sh
    ./docker_reset.sh
    ```

3. Start bash inside the Docker image:

    ```
    ./docker_bash.sh
    ```

4. INSIDE THE DOCKER IMAGE, run the dependency installation script:

    ```
    spider/michigan-private/install_dependencies.sh
    ```

5. INSIDE THE DOCKER IMAGE, `cd` into the `spider` directory and develop, run tests, etc.:

    ```
    cd spider
    ...  # Do stuff
    ```

### Other notes

* If no dependencies have been changed, you can generally just run `./docker_bash.sh` to start developing. However, you may want/need to run `./docker_build.sh` and `./docker_reset.sh` if you update the Docker image hash or project dependencies.

## Internal Tests and Pipelines

Whenever you're developing, you must make sure that our internal tests pass and that our pipelines give sensible results.

Unless otherwise stated, you should run these commands inside the Docker image from the `/spider` folder.

### Internal tests

You can run all internal tests by running this command:

```bash
python3 setup.py test
```

Make sure this command reports 0 errors and 0 failures.

`nose` provides useful features for running and debugging individual tests. Make sure it's installed:

```bash
pip3 install nose
```

To run all tests and automatically enter a debugger at the failure point, run this command:

```
nosetests --pdb
```

#### Running individual tests and debugging

Here is an example of running a specific test:

```
nosetests spider.tests.audio_featurization_test.TestAudio
```

Note that the fully-qualified name of a test can be found either from the output of `python3 setup.py test` or by following the directory structure (e.g., the above test is a class located in `/spider/tests/audio_featurization_test.py`).

To enter a debugger upon failing a specific test:

```
nosetests --pdb spider.tests.audio_featurization_test.TestAudio
```

### Pipelines

Pipelines are run on datasets provided by D3M and are used to demonstrate that our primitives can perform well on various data-driven tasks. They should be tuned to produce the highest possible scores.

Pipeline scripts can be found in `spider/pipelines`. Make sure the internal tests pass before you start testing pipelines.

#### Running pipelines

Before running one or all pipelines, you must generate the primitive and pipeline "annotations" (i.e., JSON files that describe our primitives and pipelines in D3M's format):

```bash
python3 michigan-private/genjson.py 
```

This creates three things:

* The `Michigan` folder, which contains all the annotations/JSON files
* `run_pipeline_cmds.txt`, which contains the commands used to run all pipelines and store their output scores
* The `pipeline_results` folder, which is where the output scores from the pipelines are stored

**This command must be run whenever you make changes to a pipeline.** Also note that all primitives and pipelines are associated with a brand new hash inside `Michigan` whenever this command is run. This is a D3M requirement, and makes the following tasks less intuitive.

##### Running individual pipelines

To run an individual pipeline (e.g., `VGG16HandgeometryPipeline`), look for the line for that pipeline inside `run_pipeline_cmds.txt` and run it inside your bash shell. The following command does this programmatically:

```bash
eval `cat run_pipeline_cmds.txt | grep VGG16HandgeometryPipeline`
```

This will create a `.yaml` log inside `Michigan` and a score file inside `pipeline_results`. You should aim to get the reported score as high as possible.

##### Running all pipelines in parallel

You can run all pipelines in parallel with the utility script below:

```bash
python3 michigan-private/run_pipelines.py
```

This just runs all the commands in `run_pipeline_cmds.txt` with several workers.

## Merging Changes

When you're developing a new chunk of code, please follow the general workflow outlined below. Note that the instructions cover the possibility of a pre-existing branch, but it's generally bad practice to keep them around unless you're planning to use them soon.

1. Create a new branch with a descriptive name:

    ```
    git checkout -b new-feature
    ```
    
    Or if you have a branch that you want to update, make sure it's up-to-date with `master`:
    
    ```
    git checkout new-feature
    git merge master
    ```

2. Commit your changes to the new branch:

    ```
    git add <files to add>
    git commit -m <message>
    ```
    
3. For new branches, push to the remote repo:

    ```
    git push -u origin new-feature
    ```
    
    Or for existing branches, sync with the remote repo:
    
    ```
    git pull && git push
    ```

4. Create a merge request ([link](https://github.com/dvdmjohnson/d3m_michigan_primitives/compare))

    1. Set base branch to `master`
    2. Set compare branch to your new branch
    3. Follow on-screen instructions

5. Wait for the merge request to be completed, or make changes as requested by the D3M points-of-contact (POCs)

6. Delete the new branch locally:

    ```
    git branch -d new-feature
    ```

7. Delete the new branch remotely [here](https://github.com/dvdmjohnson/d3m_michigan_primitives/branches) by clicking the appropriate Delete icon.


## [POC] Submitting to datadrivendiscovery/primitives

**Note: This section only applies to D3M's Michigan points-of-contact (POCs). If you're a normal developer for this project and want to merge changes into `master`, refer to [Merging changes into master](#merging-changes-into-master) or ask the POCs for help.**

This guide describes how to percolate changes in this repository (`dvdmjohnson/d3m_michigan_primitives` on GitHub) to the final D3M repository (`datadrivendiscovery/primitives` on GitLab). It assumes that you are trying to patch the latest commit in this repository's `master` branch into the final D3M repository.

This process establishes a connection between three repositories, each with nuanced differences:

[github.com:/dvdmjohnson/d3m_michigan_primitives](https://github.com/dvdmjohnson/d3m_michigan_primitives) -> [gitlab.com:/rszeto/primitives](https://gitlab.com/rszeto/primitives) -> [gitlab.com:/datadrivendiscovery/primitives](https://gitlab.com/datadrivendiscovery/primitives)

1. (Only required the first time you're submitting) Set up local repos

    1. On your local machine, clone the following repos:
    
        ```bash
        git clone git@github.com:dvdmjohnson/d3m_michigan_primitives.git
        git clone git@gitlab.com:rszeto/primitives.git
        ```
    
    2. Track the upstream `datadrivendiscovery/primitives` repo in your local `rszeto/primitives` repo:
    
        ```bash
        cd primitives
        git remote add d3m git@gitlab.com:datadrivendiscovery/primitives.git
        ```

2. Start a fresh Docker image and launch a bash shell inside it:

    ```bash
    cd d3m_michigan_primitives
    michigan-private/docker_build.sh
    michigan-private/docker_reset.sh
    michigan-private/docker_bash.sh 
    ```

3. Inside the Docker bash shell:

    1. Install dependencies:
    
        ```bash
        cd spider
        michigan-private/install_dependencies.sh
        ```
    
    2. Run local tests:
    
        ```bash
        python3 setup.py test
        ```
    
    3. Generate annotations/JSON files for primitives and pipelines:
    
        ```bash
        python3 michigan-private/genjson.py
        python3 michigan-private/run_pipelines.py
        ```
    
    4. Post-process the primitive/pipeline export folder (compress pipeline log runs and change permissions to host user)
    
        ```bash
        michigan-private/prepare_primitive_export.sh
        ```

4. Copy `Michigan` folder to local `rszeto/primitives` repo, and push changes:

    ```bash
    michigan-private/export_michigan_primitives.sh ../primitives
    ```

5. Ensure that the continuous integration (CI) tests in `rszeto/primitives` pass for your new branch. You can check this [here](https://gitlab.com/rszeto/primitives/pipelines) or via the email that gets sent when the tests complete. If this step fails, you must make the changes in your local `dvdmjohnson/d3m_michigan_primitives` repo and complete the entire submission process again.

5. Merge the changes in the new  `rszeto/primitives` branch into `datadrivendiscovery/primitives` via the GitLab website

    1. [Create a merge request](https://gitlab.com/rszeto/primitives/merge_requests/new) to merge your new branch in `rszeto/primitives` into `datadrivendiscovery/primitives`
    
        * The source project should be `rszeto/primitives`, and the source branch should be your new branch.
        * The target project should be `datadrivendiscovery/primitives`, and the target branch should be `master`.
        * Check off "Delete source branch when merge request is accepted."

## Other notes

Formerly, there was a note in the main README that read:

> Note that installing in development mode is currently required.

This probably refers to installing our SPIDER package in [setuptools' Development Mode](https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode). However, it's unclear if this is still necessary for either the developer or the end users.
