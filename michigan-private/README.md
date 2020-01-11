Docker Setup
-----
The current recommended method to install, test and use this library is by installing it within a pregenerated d3m docker image, which already contains up-to-date versions of the full d3m universe, including the various components that our primitives/pipelines depend on.

1. To install Docker, follow the recommended installation procedure for your OS (for Ubuntu, [installing via Docker's repository is recommended](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-using-the-repository)).

2. Go into this project's root folder and build the D3M Docker image by running the build script:

    ```
    cd spider
    michigan-private/dockerbuild.sh
    ```

3. Launch the Docker image (with the SPIDER project directory mounted within):

    ```
    michigan-private/dockerstart.sh
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


Running individual tests and debugging
-----

**NOTE: If you're using Docker, run these commands inside the Docker image.**

You can use `nose` to either run individual tests or automatically jump into a debugger on failures. `nose` must be installed as a Python package:

```
pip3 install nose
```

Here is an example of running a specific test:

```
nosetests spider.tests.audio_featurization_test.TestAudio
```

To run all tests and automatically enter a debugger at the failure point, run this command:

```
nosetests --pdb
```

Finally, to enter a debugger upon failing a specific test:

```
nosetests --pdb spider.tests.audio_featurization_test.TestAudio
```

Merging changes into master
-----

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
    git add ...
    git commit ...
    ```
    
3. For new branches, push to the remote repo:

    ```
    git push -u origin new-feature
    ```
    
    Or for existing branches, sync with the remote repo:
    
    ```
    git pull && git push
    ```

4. Create a merge request ([link](https://opticnerve.eecs.umich.edu:8888/darpa_d3m_spider/spider/merge_requests/new))

    1. Set source branch to `darpa_d3m_spider/spider`, `new-feature`
    2. Set target branch to `darpa_d3m_spider/spider`, `master`
    3. Follow on-screen instructions

5. Wait for the merge request to be completed, or make changes as requested by the D3M points-of-contact (POCs)

6. Delete the new branch locally:

    ```
    git branch -d new-feature
    ```

7. Delete the new branch remotely ([here](https://opticnerve.eecs.umich.edu:8888/darpa_d3m_spider/spider/branches)) by clicking the appropriate Delete icon.


Submitting to datadrivendiscovery/primitives
-----

**Note: This section only applies to D3M's Michigan points-of-contact (POCs). If you're a normal developer for this project and want to merge changes into `master`, refer to [Merging changes into master](#merging-changes-into-master) or ask the POCs for help.**

This guide describes how to percolate changes in this repository (`darpa_d3m_spider` on opticnerve) to the final D3M repository (`datadrivendiscovery/primitives` on GitLab). It assumes that you are trying to patch the latest commit in this repository's `master` branch into the final D3M repository.

This process establishes a connection between a whopping four repositories, each with nuanced differences:

[opticnerve.eecs.umich.edu:/darpa_d3m_spider/spider](https://opticnerve.eecs.umich.edu:8888/darpa_d3m_spider/spider) -> [github.com:/dvdmjohnson/d3m_michigan_primitives](https://github.com/dvdmjohnson/d3m_michigan_primitives) -> [gitlab.com:/rszeto/primitives](https://gitlab.com/rszeto/primitives) -> [gitlab.com:/datadrivendiscovery/primitives](https://gitlab.com/datadrivendiscovery/primitives)

1. (Only required the first time you're submitting) Set up local repos

    1. On your local machine, clone the following three repos:
    
        ```bash
        git clone git@opticnerve.eecs.umich.edu:darpa_d3m_spider/spider.git
        git clone git@github.com:dvdmjohnson/d3m_michigan_primitives.git
        git clone git@gitlab.com:rszeto/primitives.git
        ```
    
    2. Track the upstream `datadrivendiscovery/primitives` repo in your local `rszeto/primitives` repo:
    
        ```bash
        cd primitives
        git remote add d3m git@gitlab.com:datadrivendiscovery/primitives.git
        ```

2. Create a patch branch in the `dvdmjohnson/d3m_michigan_primitives` repo

    1. `cd` into the `dvdmjohnson/d3m_michigan_primitives` local repo and update the master branch

        ```bash
        cd d3m_michigan_primitives
        git checkout master && git pull
        ```

    2. `cd` into the `darpa_d3m_spider/spider` local repo and update the master branch

        ```bash
        cd spider
        git checkout master && git pull
        ```

    3. Run the following script in this `darpa_d3m_spider/spider`, which creates and pushes a patch branch to `dvdmjohnson/d3m_michigan_primitives` based on the latest commit in this `darpa_d3m_spider/spider` repo:

        ```bash
        michigan-primitives/create_github_commit.sh <path to local d3m_michigan_primitives repo>
        ```

3. Merge the patch branch of `dvdmjohnson/d3m_michigan_primitives` into `master`

    1. If you're not a master member of `dvdmjohnson/d3m_michigan_primitives`, [you must create a pull request on GitHub](https://github.com/dvdmjohnson/d3m_michigan_primitives/pulls).
    
    2. If you're a master member of `dvdmjohnson/d3m_michigan_primitives`, you can do the merge locally and push to remote:
    
        ```bash
        cd d3m_michigan_primitives
        git checkout master && git pull
        git merge <patch_branch_name>
        git push
        ```

4. Export the primitives from `dvdmjohnson/d3m_michigan_primitives` on GitHub to `rszeto/primitives` on GitLab

    1. Pull the latest changes from `datadrivendiscovery/primitives` into `rszeto/primitives`, and push them to the remote repo:
    
        ```bash
        cd primitives
        # Update local copy of `rszeto/primitives`
        git checkout master && git pull
        # Merge changes from upstream datadrivendiscovery/primitives repo into local repo
        git fetch d3m && git merge d3m/master
        # Push latest changes into remote rszeto/primitives repo
        git push
        ```
    
        This essentially updates `rszeto/primitives` to be in sync with `datadrivendiscovery/primitives` (both remotely and locally).

    2. In the `darpa_d3m_spider/spider` local repo, launch the Docker image

        ```bash
        cd spider
        michigan-primitives/dockerbuild.sh
        michigan-primitives/dockerstart.sh
        ```

    3. Inside the Docker image, run the following script, which pushes a patch branch to `rszeto/primitives` based on the latest `dvdmjohnson/d3m_michigan_primitives` commit:

        ```bash
        spider/michigan-primitives/export_michigan_primitives.sh
        ```

        This command will ask for your GitLab credentials at the end and possibly your D3M credentials at the beginning. If you enter the incorrect credentials, the new branch will not be created properly. If this happens, go back to the previous step (i.e., restart the Docker image) and try again.

    4. Ensure that the continuous integration (CI) tests in `rszeto/primitives` pass for your new branch. You can check this [here](https://gitlab.com/rszeto/primitives/pipelines) or via the email that gets sent when the tests complete. If this step fails, you must make the changes in your local `darpa_d3m_spider/spider` repo and complete the entire submission process again.

5. Merge the changes in the new  `rszeto/primitives` branch into `datadrivendiscovery/primitives` via the GitLab website

    1. [Create a merge request](https://gitlab.com/rszeto/primitives/merge_requests/new) to merge your new branch in `rszeto/primitives` into `datadrivendiscovery/primitives`
    
        * The source project should be `rszeto/primitives`, and the source branch should be your new branch.
        * The target project should be `datadrivendiscovery/primitives`, and the target branch should be `master`.

    2. Once the merge request is complete, update the local and remote `rszeto/primitives` repos:
    
        ```bash
        cd primitives
        # Update local copy of `rszeto/primitives`
        git checkout master && git pull
        # Merge changes from upstream datadrivendiscovery/primitives repo into local repo
        git fetch d3m && git merge d3m/master
        # Push latest changes into remote rszeto/primitives repo
        git push
        ```

Other notes
-----

Formerly, there was a note in the main README that read:

> Note that installing in development mode is currently required.

This probably refers to installing our SPIDER package in [setuptools' Development Mode](https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode). However, it's unclear if this is still necessary for either the developer or the end users.
