### Low-Light-Image-Enhancement-by-Filters-and-Frequency-Domain-Representations
This repo is solely used for recording the project reproducing and modifications making. Not for any publication or commercial use.
### Our Approach
1. Propose LLIE F2, an end-to-end framework that integrates registered infrared information to guide the restoration of visible low-light images, effectively solving the problem of structural information loss in extreme darkness.
2. seamlessly incorporate interpretable digital filters (Butterworth, IIR, and Adaptive FIR) into deep neural networks. To the best of our knowledge, this is one of the first attempts to use explicit DSP theories to stabilize frequency-domain features and spatial texture responses in LLIE tasks.
3. Designadual-stage architecture that decouples global illumination correction (via Amplitude Phase modulation) from local detail refinement, ensuring global consistency while maximizing
local texture fidelity.
4. Extensive experiments on the LOL-v2-Real, LOL-v2-Syn, and LSRW-Huawei datasets demonstrate that our approach achieves state-of-the-art performance in both quantitative metrics and visual quality.
### Acknowledgement
This repo is based on [DMFourLLIE](https://github.com/bywlzts/DMFourLLIE)
