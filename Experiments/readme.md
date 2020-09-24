
# Astro_U-net
<p align="center"> <b>  Astro-U-net:  Learning to Enhance Astronomical Images </b> </p>


<p style="text-align:justify"> Experiments are fun! During training of our networks we work with different hyperparameters to find the best ones. In this section we provide the results from different networks which we trained. We evalueted our networks with the  <a href="http://munipack.physics.muni.cz/">Munipack</a> and the <a href = " https://www.astromatic.net/software/sextractor">Source Extractor</a>. First table contains results from Source Extractor, second one from the Munipack and in the last one reader can find the distribution results. </p>


  |                          Source Extractor                                |||||||
  | --- | --- | --- | --- | --- | --- |
  |Network | Flux error | True positive |	True positive rate |	F-measure| SNR |
  | --- | --- | --- | --- | --- | --- |
  | Just L1 | --- | --- | --- | --- | --- |
  | Just L2 | --- | --- | --- | --- | --- |
  |Segmentation map | --- | --- | --- | --- | --- |
  |L1 + KL divergence loss  | --- | --- | --- | --- | --- |
  |Input multiplied by ETR, L1 loss | --- | --- | --- | --- | --- |
  | Network 1 | --- | --- | --- | --- | --- |
  | L1 + KL divergence loss + ETR on the bottom| --- | --- | --- | --- | --- |
  | Multiple ETR (random order) | --- | --- | --- | --- | --- |
  | Network 2 | --- | --- | --- | --- | --- |
  | ReLU activation | --- | --- | --- | --- | --- |
  | PReLU activation | --- | --- | --- | --- | --- |
  | Swish activation | --- | --- | --- | --- | --- |
  | Input Image | --- | --- | --- | --- | --- |
  

  |Network | Flux error | True positive |	True positive rate |	F-measure| SNR |
  | --- | --- | --- | --- | --- | --- |
  | Just L1 | --- | --- | --- | --- | --- |
  | Just L2 | --- | --- | --- | --- | --- |
  |Segmentation map | --- | --- | --- | --- | --- |
  |L1 + KL divergence loss  | --- | --- | --- | --- | --- |
  |Input multiplied by ETR, L1 loss | --- | --- | --- | --- | --- |
  | Network 1 | --- | --- | --- | --- | --- |
  | L1 + KL divergence loss + ETR on the bottom| --- | --- | --- | --- | --- |
  | Multiple ETR (random order) | --- | --- | --- | --- | --- |
  | Network 2 | --- | --- | --- | --- | --- |
  | ReLU activation | --- | --- | --- | --- | --- |
  | PReLU activation | --- | --- | --- | --- | --- |
  | Swish activation | --- | --- | --- | --- | --- |
  | Input Image | --- | --- | --- | --- | --- |
