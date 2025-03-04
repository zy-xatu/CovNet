<p align="center">

  <h1 align="center">CovNet: Leveraging Covariance Features for Accurate Infrared Small Target Detection</h1>
  <div align="center">
    <img src="video.gif", width="1200">
  </div>
</p>

## :bookmark: Manuscript
Our manuscript has been formally submitted to [The Visual Computer](https://link.springer.com/journal/371) journal for peer review and publication consideration.

## :whale: CheckPoints and Model Files
We provide all checkpoints and models at [here](https://drive.google.com/drive/folders/1yy_BZMGZuQQVrw21r7qTtZaTkT_AdyU3?usp=drive_link).

## :sparkles: Our Test Code
## Dependencies and Download

- NVIDIA GPU + CUDA >=11.2
- Linux Ubuntu 18.04

## Requirements
Then install the additional requirements
```
pip install -r requirements.txt
```

## Download pre-trained weights
Download dataset and model weights. Then put them in the folder. The hierachy is as follows:
![image](https://github.com/user-attachments/assets/d99db6fc-53af-42ce-8b2b-e26719bda176)

## RUN code
```
python train.py
```
```
python detect.py
```
## Acknowledgement
The dataset and validation set used in our experiment are from [here](http://www.csdata.org/p/387/). Our training and validation sets and corresponding labels can be obtained [here](https://drive.google.com/file/d/1WQacfRBWribfKicJO80gjBJUYpvqNc-k/view?usp=drive_link).
