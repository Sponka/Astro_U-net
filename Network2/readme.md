 <p align="center"> <b>  Astro-U-net:  Network 2 </b> </p>
 
 <p style="text-align:justify"> In our second network, we trained on exposure time ratios between two and five. The ratio is selected in order -- for every image crop we trained network with all ratios (2,3,4,5). Network 2 is trained for 3000 epochs, with 150 iterations per epoch. In each  iteration the network sees the same random image crop, with all exposure time ratios in order. Training of the network takes approximately 62 hours on NVIDIA GTX1080 Ti</p>
 
 
 
 | Flux error | True positive |	True positive rate |	F-measure| SNR | PSNR | SSIM | KL [10-7]|
 | --- | --- | --- | --- | --- | --- | --- | --- |
 | 0.0| 0.0 | --- | --- | --- | --- | --- | --- |
