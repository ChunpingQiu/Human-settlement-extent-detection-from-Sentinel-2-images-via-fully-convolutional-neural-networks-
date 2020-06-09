# Use the trained model to prodict HSE from S2 images
## use demo
- put the image data (s2) into folder: ./data/img/  
- run python img2map.py
- prediciton will be in folder ./data/pre/

## reference data
https://drive.google.com/drive/folders/1n2LGeGAv_O2cvxAJnSGNRUI4FMsm4psa?usp=sharing

## current status
I am going to focus regression of HSE, instead of the binary classification. 
The framework is similar as this one.

So I will create a new project and upload data and code there.

## further information

```
@article{qiuFCN,
  title={A framework for large-scale mapping of human settlement extent from Sentinel-2 images via fully convolutional neural networks},
  author={Qiu, Chunping and Schmitt, Michael and Gei{\ss}, Christian and Chen, Tzu-Hsin Karen and Zhu, Xiao Xiang},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={163},
  pages={152--170},
  year={2020},
  publisher={Elsevier}
}
```
