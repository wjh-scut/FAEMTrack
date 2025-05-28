# FAEMTrack
## FAEMTrack: Feature-Augmented Embedding and Cross-Drone
Fusion for Single Object Tracking
In this project, we introduce FAEMTrack, a state-of-the-art multi-drone tracking framework designed to tackle occlusion, viewpoint variation, and target disappearance in challenging environments. FAEMTrack integrates a Feature-Augmented Embedding Module (FAEM)—which replaces standard convolutions with multi-scale depthwise convolution blocks (MSDB) and a spatial attention (SA) mechanism to enrich spatial–semantic encoding—and an entropy-weighted cross-drone fusion strategy that dynamically balances per-drone response confidence for adaptive, collaborative tracking. Extensive experiments on the MDOT benchmark demonstrate that FAEMTrack achieves superior accuracy and robustness compared to existing single- and multi-drone methods.  
<div  align="center">    
 <img src="./figure/framework.jpg" width = "777"  align=center />
</div>
