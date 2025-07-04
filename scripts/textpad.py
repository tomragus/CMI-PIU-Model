"""

Training data shape: (3960, 82)
Test data shape: (20, 59)
After cleaning - Training: (2736, 81), Test: (20, 59)

=== MISSING DATA ANALYSIS ===
Top 15 features with most missing data:
                              Train_Missing_%  Test_Missing_%
PAQ_A-Season                        86.732456            95.0
PAQ_A-PAQ_A_Total                   86.732456            95.0
Physical-Waist_Circumference        82.346491            75.0
Fitness_Endurance-Time_Sec          73.391813            85.0
Fitness_Endurance-Time_Mins         73.391813            85.0
Fitness_Endurance-Max_Stage         73.282164            85.0
FGC-FGC_GSND_Zone                   68.421053            75.0
FGC-FGC_GSD_Zone                    68.421053            75.0
FGC-FGC_GSD                         68.165205            75.0
FGC-FGC_GSND                        68.128655            75.0
Fitness_Endurance-Season            53.947368            80.0
PAQ_C-Season                        47.368421            55.0
PAQ_C-PAQ_C_Total                   47.368421            55.0
BIA-BIA_TBW                         33.735380            60.0
BIA-BIA_BMC                         33.735380            60.0

Features with >80% missing data (3): ['PAQ_A-Season', 'PAQ_A-PAQ_A_Total', 'Physical-Waist_Circumference']

=== FEATURE GROUPS ===
demographics: 3 features
physical_basic: 7 features
fitness: 3 features
functional_tests: 15 features
body_composition: 17 features
activity_questionnaires: 4 features
psychological: 22 features
other_assessments: 5 features
education: 2 features

=== FEATURE ENGINEERING ===
Created 14 engineered features:
['Age_Group', 'PAQ_A-PAQ_A_Total_missing', 'PCIAT_Control', 'BMI_Age_Adjusted', 'Waist_Height_Ratio', 'BP_Category', 'PCIAT-PCIAT_Total_missing', 'PCIAT_Compulsive', 'Fat_to_FFM_Ratio', 'PCIAT_Neglect', 'Fitness_Index', 'Physical-BMI_missing', 'Activity_Screen_Balance', 'BMI_category']

=== MISSING VALUE IMPUTATION ===
Missing value imputation completed!
Remaining missing values in train: 0
Remaining missing values in test: 0

=== FEATURE SELECTION (mutual_info) ===
Selected 40 features out of 95
Top 10 selected features:
  PCIAT-PCIAT_Total: 0.990
  PCIAT_Control: 0.695
  PCIAT_Compulsive: 0.548
  PCIAT_Neglect: 0.449
  PCIAT-PCIAT_15: 0.362
  PCIAT-PCIAT_03: 0.361
  PCIAT-PCIAT_05: 0.351
  PCIAT-PCIAT_17: 0.348
  PCIAT-PCIAT_02: 0.333
  PCIAT-PCIAT_18: 0.309

=== FEATURE SCALING (robust) ===
Feature scaling completed!

"""