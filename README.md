# CMPT 985 Final Project
### Xiaohao Sun, Yongsen Mao, Yiming Zhang


This project benchmarks different 3D reconstruction methods with monocular RGB images. The input is a set of monocular RGB images from different viewpoints of the scene and the camera parameters. Some approaches may also use depth information as part of the input for better performance. With these inputs, the output is supposed to be a 3D scene geometry of the specific scene. There are three different kinds of methods: MonoSDF, NeuralRecon, and MVS2D.

For instructions for data proprocessing and experiments, please go to the submodules for three methods and refer to their READMEs.

To evaluate three methods, please run:
`python -p <path_to_predicted_mesh> -g <path_to_GT_mesn>`
Then, Precision, Recall, F1-score and Chamfer distance will be reported.