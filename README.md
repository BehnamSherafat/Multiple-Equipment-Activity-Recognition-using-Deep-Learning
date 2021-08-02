# MultiEqpActivityRecognition
**Note 1:**
All rights reserved. Please cite at least one of these papers if you have used the dataset and code in this project:
1) Sherafat, B., Rashidi, A., Lee, Y.C. and Ahn, C.R., 2019. Automated activity recognition of construction equipment using a data fusion approach. In Computing in civil engineering 2019: Data, sensing, and analytics (pp. 1-8). Reston, VA: American Society of Civil Engineers.
2) Sherafat, B., Rashidi, A., Lee, Y.C. and Ahn, C.R., 2019. A hybrid kinematic-acoustic system for automated activity detection of construction equipment. Sensors, 19(19), p.4286.
3) Sherafat, B., Ahn, C.R., Akhavian, R., Behzadan, A.H., Golparvar-Fard, M., Kim, H., Lee, Y.C., Rashidi, A. and Azar, E.R., 2020. Automated methods for activity recognition of construction workers and equipment: state-of-the-art review. Journal of Construction Engineering and Management, 146(6), p.03120002.
4) Sherafat, B., Rashidi, A. and Song, S., 2020, November. A Software-Based Approach for Acoustical Modeling of Construction Job Sites with Multiple Operational Machines. In Construction Research Congress 2020: Computer Applications (pp. 886-895). Reston, VA: American Society of Civil Engineers.
5) Sherafat, B., Rashidi, A. and Asgari, S., 2020, December. Comparison of different beamforming-based approaches for sound source separation of multiple heavy equipment at construction job sites. In 2020 Winter Simulation Conference (WSC) (pp. 2435-2446). IEEE.


# A Sound-based Multiple-Equipment Activity Recognition Model
This model uses construction equipment sound files (.wav) to train a model for multiple-equipment activity recognition.
The recorded sound of each equipment is pre-processed (e.g., de-noised) and used in a data augmentation process to generate synthetic mix.
The generated synthetic mixes are used to train a Convolutional Neural Network (CNN) for multiple-equipment activity recognition.
There are 5 different case studies, each in a seperate folder:
1) **Synthetic Mix 1**
2) **Synthetic Mix 2**
3) **Real-World Mix 1**
4) **Real-World Mix 2**
5) **Real-World Mix 3**

The first two folders are synthetic mixes. In other worlds, individual recorded sounds for each equipment is mixed using "S_Mixed = S1 + S2" equation in the data augmentation process. The next three folders contain real-world mixes, which are captured in real construction job sites when multiple machines were performing activities simultaneously. 
Also, there are 3 more folders which are considered for comparison with 2 CNN baseline models:
1) MobileNetV2
2) ResNet18

To run the codes, several steps should be considered:
1) Open and run the "Data Augmentation" file for each case study (i.e., folder).
2) Run the training for each equipment:
  - Open and run "TrainActRec" file for equipment No. 1.
  - Open and run "TrainActRec2" file for equipment No. 2.
3) Run the testing for each equipment:
  - Open and run "TestActRec" file for equipment No. 1.
  - Open and run "TestActRec2" file for equipment No. 2.
 
 **Note 2:** The augmented dataset is not uploaded here due to its large size (more than 15 Gb). The "Data Augmentation" file generates this dataset and its corresponding dataset.
 **Note 3:** The "-ori" suffix shows that this sound file is the original sound file (i.e., no de-noising is conducted).
 **Note 4:** The "-den" suffix shows that this sound file is denoised using "Rangachari, S. and Loizou, P.C., 2006. A noise-estimation algorithm for highly non-stationary environments. Speech communication, 48(2), pp.220-231. 10.1016/j.specom.2005.08.005" proposed algorithm.
 
 
