# Data Folder

This folder contains three `.zip` files:

- `fingerprints.zip`, a collection of the fingerprints collected from the target devices that were used in simulating
  the reinforcement learning (RL) environment.
- `additional_benign_behavior_fingerprints.zip`, the collection of fingerprints which were collected while executing
  additional benign behaviors on the victim devices.
- `plots_and_results.zip`, a collection of all accuracy computations, metrics/results, and plots that were used
  throughout the evaluations addressed in the report.

Due to some of the files exceeding GitHub's file size limit (100 MB), these `.zip` files are tracked
by [Git LFS](https://git-lfs.com/).
There is also
a [documentation on GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github),
including the same setup instructions and some additional information on why and how to use Git LFS.

- **DISCLAIMER:** _If you don't have Git LFS installed already, you need to do so to open the collections. Moreover, you
  will need to remove and re-clone this repository because the large files are not properly cloned by Git when Git LFS
  is not installed (
  see [documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/collaboration-with-git-large-file-storage))._
