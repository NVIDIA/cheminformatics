# Cheminformatics
A useful task in drug discovery is searching, screening, and organizing large chemical databases. This cheminformatics application demonstrates real-time exploration and analysis of a database of chemical compounds. Molecules are clustered based on chemical similarity and visualized with an interactive plot. Users are able to explore in real time regions of interest in chemical space and see the corresponding chemical structures and physical properties. Users can also generate new molecules either by exploring the latent space between two molecules or sampling around a molecule. In the current version user can select between two generative models. The two generative models are:
 - [CDDD](https://github.com/jrwnter/cddd)
 - [Mega MolBART](https://ngc.nvidia.com/models/nv-drug-discovery-dev:megamolbart)

Using RAPIDS, compounds in the ChEMBL database are clustered on molecular similarity. We compute Morgan fingerprints, cluster using k-means, and perform dimensionality reduction using UMAP. Distributed GPU-accelerated algorithms enable real-time interactive exploration of chemical space.

[Mega MolBART](https://ngc.nvidia.com/models/nv-drug-discovery-dev:megamolbart) cannot be used in stand-alone mode. Please use the [Cheminformatics](https://ngc.nvidia.com/resources/nv-drug-discovery-dev:cheminformatics) resource for complete suite.

# Getting Started
The demo can be launched with a single docker run command. Before doing so, it is helpful to identify a directory where the ChEMBL database can be downloaded. The container will automatically download the database for you, regardless of whether this directory is specified. However, if you would like to re-launch the container, specifying this directory will save time by not re-downloading the database if it is already present.

In the launch command below, we assume that a directory /tmp exists and can be used by the container to download the database. Launch the container by:

```
docker run --gpus all -it -v /tmp/:/data -p 5000:5000 nvcr.io/nvidia/clara/cheminformatics_demo:0.1
```

Watch the terminal output for an indication of download progress. Once the download is complete, the dash application will initialize and the following text will indicate that the application is ready:

```
INFO:werkzeug: * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
```

Open a web browser to localhost:5000 to begin exploring the ChEMBL database interactively.

# Tutorial
See the [tutorial](https://github.com/NVIDIA/cheminformatics/blob/master/tutorial/Tutorial.md) for an example walkthrough.

# Features
- Cluster molecules from ChEMBL using the embedding generated from Morgan Fingerprints --> PCA --> UMAP
- Ability to color the clustered molecules based on molecular properties
- Ability to recluster on user selected subsets of molecules or specific clusters
- Designate and track molecules of interest during the analysis
- Generate new molecules by linearly interpolating the latent space between two selected molecules or sampling arround a selected molecule
- Export generated molecules in SDF format