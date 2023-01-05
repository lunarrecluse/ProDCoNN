# ProDCoNN
Designing protein sequences that fold to a given three-dimensional (3D) structure has long been a challenging problem in computational structural biology with significant theoretical and practical implications. In this study, we first formulated this problem as predicting the residue type given the 3D structural environment around the Cα atom of a residue, which is repeated for each residue of a protein. We designed a nine-layer 3D deep convolutional neural network (CNN) that takes as input a gridded box with the atomic coordinates and types around a residue. Several CNN layers were designed to capture structure information at different scales, such as bond lengths, bond angles, torsion angles, and secondary structures. Trained on a huge number of protein structures, the method, called ProDCoNN (protein design with CNN), achieved state-of-the-art performance when tested on large numbers of test proteins and benchmark datasets.

This package provides an implementation of the inference pipeline of ProDCoNN:
(1) 4 pre-trained models using different setups and databases. We recommend using BBO90 and BBS90. Please refer to the ProDCoNN paper for a detailed description of the method.
(2) Script for running the inference pipeline
(3) requirements.txt contains the env information 

To run the pipeline:
(1) setup the environment based on requirements.txt
(2) BBO30: python prediction.py pdb_file_path start_index, end_index BBO 30 output_file
(3) BBO90: python prediction.py pdb_file_path start_index, end_index BBO 90 output_file
(4) BBS30: python prediction.py pdb_file_path start_index, end_index BBS 30 output_file
(5) BBS90: python prediction.py pdb_file_path start_index, end_index BBS 90 output_file
pdb_file_path: the path to the input pdb file
start_index: the start index of the prediction along the sequence, count from 1
end_index: the end index of the prediction along the sequence. If all the residues need to be predicted, the start_index should be 1, and the end_index should be -1.
output_file: the name for the output file

Example:
python prediction.py 1ete.pdb A 1,-1 BBO 90 result_bbo90.txt
