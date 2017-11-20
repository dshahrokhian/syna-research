<img src="https://user-images.githubusercontent.com/8654460/32914363-9a58350e-cb15-11e7-9628-94ef6de534bf.png" width="300">

Emotion Recognition based on Spatio-Temporal Machine Learning. You can find more information about the project in my Master's [thesis](http://kth.diva-portal.org/smash/record.jsf?aq2=%5B%5B%5D%5D&c=37&af=%5B%5D&searchType=LIST_LATEST&sortOrder2=title_sort_asc&query=&language=en&pid=diva2%3A1149133&aq=%5B%5B%5D%5D&sf=all&aqe=%5B%5D&sortOrder=author_sort_asc&onlyFullText=false&noOfRows=50&dswid=5833).

## Installation [Ubuntu 16.04 LTE]
1. Run `sudo gedit /etc/apt/sources.list` and change the line `deb http://us.archive.ubuntu.com/ubuntu/ xenial main restricted` to `deb http://us.archive.ubuntu.com/ubuntu/ xenial main universe`

2. Run `install.sh`

## Repository Structure
- **/data** contains auxiliary data for training and processing
- **/datasets** is expected to contain the datasets necessary for training syna. The datasets so far are *The Extended Cohn-Kanade (CK+) database* and *Acted Facial Expressions In The Wild database 6.0*. Given that I have no rights to publish those datasets, you'll have to contact the authors for download
- **/experiments** contains a set of experiments to test the developed models, together with a set of utils
- **/scripts** contains useful scripts to parse the datasets before applying the syna and deepsyna models. Make sure that the datasets are contained in the *datasets* directory before executing
- **/syna** contains the main codebase of the project: C3D, dataloaders, face-frontalization, a custom version of OpenFace, and LSTM.

## Citation
If you use any of the resources provided on this page in any of your publications, I ask you to cite the following work:

**Syna: Emotion Recognition based on Spatio-Temporal Machine Learning**
Daniyal Shahrokhian, *KTH Royal Institute of Technology*, 2017

## Open Source Acknowledgements

[OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)

[C3D Keras](https://github.com/axon-research/c3d-keras)

[Face Frontalization](https://github.com/ChrisYang/facefrontalisation)

## References

[Constrained Local Neural Fields for robust facial landmark detection in the wild](https://www.cl.cam.ac.uk/~tb346/pub/papers/iccv2013.pdf)

[Learning Spatiotemporal Features with 3D Convolutional Networks](http://vlg.cs.dartmouth.edu/c3d/c3d_video.pdf)

[Effective Face Frontalization in Unconstrained Images](http://www.openu.ac.il/home/hassner/projects/frontalize)
