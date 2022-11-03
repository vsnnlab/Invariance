### Demo codes for
### "Invariance of object detection in untrained deep neural networks" </br>

Jeonghwan Cheon, Seungdae Baek, and Se-Bum Paik*

*Contact: sbpaik@kaist.ac.kr

### 1. System requirements
- MATLAB 2021b or later version
- Installation of the Deep Learning Toolbox (https://www.mathworks.com/products/deep-learning.html)
- Installation of the pretrained AlexNet (https://de.mathworks.com/matlabcentral/fileexchange/59133-deep-learning-toolbox-model-for-alexnet-network)
- No non-standard hardware is required.
- Uploaded codes were tested using MATLAB 2021b.

### 2. Installation
- Download all files and folders. ("Clone or download" -> "Download ZIP")
- Download 'Image.zip' from below link and unzip files in the same directory
- Dataset Download: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7276304.svg)](https://doi.org/10.5281/zenodo.7276304)
- Expected Installation time is about 60 minutes, but may vary by system conditions.
 
### 3. Instructions for demo
- Edit "Main.m' to select the class of the object-selective units to which perform analysis.
- Select result numbers (from 1 to 5) that you want to perform a demo simulation.
- Expected running time is about 5 minutes for each figure, but may vary by system conditions.

### 4. Expected output for demo
- Below results for untrained AlexNet will be shown.
  + ``` Run_Selectivity.m ```: Emergence of selectivity to various objects in untrained networks (Result 1)
  + ``` Run_Invariance.m ```: Viewpoint-invariant object selectivity observed in untrained networks (Result 2a)
  + ``` Run_PFI.m ```: Viewpoint-invariant unit and specific units and its visual feature encoding (Result 2b)
  + ``` Run_Connectivity.m ```: Computational model explains spontaneous emergence of invariance in untrained networks (Result 3)
  + ``` Run_SVM.m ```: Invariantly tuned unit responses enable invariant object detection (Result 4)

### 5. Citation
```bibtex
@ARTICLE{10.3389/fncom.2022.1030707,
  AUTHOR={Cheon, Jeonghwan and Baek, Seungdae and Paik, Se-Bum},
  TITLE={Invariance of object detection in untrained deep neural networks},
  JOURNAL={Frontiers in Computational Neuroscience},
  VOLUME={16},
  YEAR={2022},
  URL={https://www.frontiersin.org/articles/10.3389/fncom.2022.1030707},
  DOI={10.3389/fncom.2022.1030707},
  ISSN={1662-5188}
}
```
