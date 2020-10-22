
 <p align="center"> <b>  Astro-U-net:  Network 1 </b> </p>
 
 <p style="text-align:justify">  The input and output have one channel with size 256x256. Information about the exposure time ratio, is added at the bottom of the U-net. For training we use an exposure time ratio of two. The network is trained for 5000 epochs and it take ~ 48 hours. Each epoch has 160 iterations, where it sees a random crop from each of the 160 training images. Evaluation was done on images in electrons. We created two types of table, first one refers to cross-match SExtractor files for Ground Truth x Output and the second one refers to Ground Truth x Output x Input. PSNR, SSIM, KL are same for both tables and the SNR is calculated just for Ground Truth x Output x Input. </p>
 
 
<br/><br/>
 
 |Image| RFE [%] | RFE error [%] | TP |TPR [%] |F-measure| SNR | PSNR | SSIM | KL|
 | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
 |Network Output | 2.26|0.18| 4334 | 98.94 | 0.86 | 1.63 | 13.6 | 0.64 | 0.007 |
 |Input | 1.55| 0.21 | 3366 | 68.08 | 0.78 | 0 | -16 | 0.45 | 0.0231 |
  
  Cross-match of Ground Truth x Output.  Relative flux error is denote as RFE, True Positive as TP and True Positive Rate as TPR.

<br/><br/>
 |Image|Ratio | RFE [%] | RFE error [%] | TP | TPR [%] | SNR | 
 | --- | --- | --- | --- | --- | --- | --- |
 |Network Output | 2 | 1.67 |0.14| 3347 | 67.43 | 1.63 |
 |Input | 2 | 1.55 | 0.21 | 3347 | 67.43 |  0 | 
 
 
 
 Cross-match of Ground Truth x Output x Input. Relative flux error is denote as RFE, True Positive as TP and True Positive Rate as TPR.
 <br/><br/>

<p align="center"><img src="eval_train_loss_net1.png" height="500px"></p>



<p align="center"> <b>  Histogram </b> </p>

	
<p align="left"><img src="hist/example1.png" height="300px"> <img src="hist/example2.png" height="300px"></p>

<p align="left"><img src="hist/example3.png" height="300px"> <img src="hist/example4.png" height="300px"></p>


<p align="center"> <b>  Residuals </b> </p>


<p align="left"><img src="Residuals/1.png" height="250px">    <img src="Residuals/histogram_1.png" height="300px"></p>

<p align="left"><img src="Residuals/6.png" height="300px">    <img src="Residuals/histogram_6.png" height="300px"></p>
