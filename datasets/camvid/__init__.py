# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
'''
Preprocessing and Data loader for the CamVid [1,2] dataset.
The code assumes the dataset has the structure of Kaggle's version of CamVid [3].

[1] Segmentation and Recognition Using Structure from Motion Point Clouds, ECCV 2008 ([pdf](http://www.inf.ethz.ch/personal/gbrostow/ext/MotionSegRecECCV08.pdf))
Brostow, Shotton, Fauqueur, Cipolla ([bibtex](http://www.cs.ucl.ac.uk/staff/G.Brostow/bibs/RecognitionFromMotion_bib.html))

[2] Semantic Object Classes in Video: A High-Definition Ground Truth Database ([pdf](http://www.cs.ucl.ac.uk/staff/G.Brostow/papers/SemanticObjectClassesInVideo_BrostowEtAl2009.pdf))
Pattern Recognition Letters ([to appear](http://www.cs.ucl.ac.uk/staff/G.Brostow))
Brostow, Fauqueur, Cipolla ([bibtex](http://www.cs.ucl.ac.uk/staff/G.Brostow/bibs/RecognitionFromMotion_bib.html))

[3] https://www.kaggle.com/datasets/carlolepelaars/camvid
'''
