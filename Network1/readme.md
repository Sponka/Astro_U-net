
 <p align="center"> <b>  Astro-U-net:  Network 1 </b> </p>
 
 <p style="text-align:justify">  The input and output have one channel with size 256x256. Information about the exposure time ratio, is added at the bottom of the U-net. For training we use an exposure time ratio of two. The network is trained for 3000 epochs and it take 37 hours on NVIDIA GTX1080 Ti. Each epoch has 160 iterations, where it sees a random crop from each of the 160 training images.  Validation of images are done after every thousand epochs and the number of epochs with the best results is chosen.</p>
 
 
 
 | Flux error | True positive |	True positive rate |	F-measure| SNR | PSNR | SSIM | KL [10-7]|
 | --- | --- | --- | --- | --- | --- | --- | --- |
 | 0.0| 0.0 | --- | --- | --- | --- | --- | --- |
