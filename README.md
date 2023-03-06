


<!-- PROJECT LOGO -->
<br />
<div align="center">
  
  <h3 align="center">Object Detection Using YOLO v3 </h3>

</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
        <li><a href="#built-with">Built With</a></li>
    </li>
        <li><a href="#installation">Installation</a></li>


  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
<div align="center">
  <img src="https://github.com/RomanRes/Object-detection-tool-based-on-Dash-and-YOLOv3/blob/main/img/YOLOv3readme.gif" >
</div>

An object detection tools. Based on YOLOv3 with pretrained weights on COCO data set. Can detect 80 classes.





### Built With


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![Dash](https://img.shields.io/badge/Dash-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![Bootstrap](https://img.shields.io/badge/bootstrap-%23563D7C.svg?style=for-the-badge&logo=bootstrap&logoColor=white)



### Installation

1. Clone the repo
```sh
git clone https://github.com/RomanRes/Object-detection-tool-based-on-Dash-and-YOLOv3.git
```
2. Create a fresh venv (with `conda` or `virtualenv`) and activate it.

3. Install the requirements:
```
pip install -r requirements.txt
```
4. Load weights and put in main folder

   https://pjreddie.com/media/files/yolov3.weights

5. Start the app:

```
python app.py
```
6. Open in browser `http://127.0.0.1:8050/`



<!-- USAGE EXAMPLES -->
## Usage

The app allows you to change the maximum value of the IOU between boxes of the same class, as well as the value of the probability of finding an object in a box, to eliminate unnecessary boxes.



