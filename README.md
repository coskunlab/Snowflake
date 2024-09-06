# Snowflake

This folder contains the scripts and data to reproduce the result in the paper "Spatial Morphoproteomic Features Predict Disease States from Tissue Architectures".

You can find the raw data here:___

![Alt text](figures/Figure2.png)

(A) Raw data obtained from multiplex imaging showing CD20 (Blue), CD21 (yellow), and Ki67 (magenta) staining from tonsil and adenoid tissues with and without COVID infection. Follicles and corresponding germinal centers are segmented using multiplex panels (n=930 follicles and n=775 germinal centers). Single cells are segmented (n=8879749 cells), and cell neighborhood graphs are extracted based on cell spatial location. (B) SNOWFLAKE prediction pipeline combining morphological information with single-cell data. The two modes of the SNOWFLAKE pipeline are the SNOWFLAKE with MorphPCA and position-aware SNOWFLAKE. In SNOWFLAKE with morphPCA, the morphological information and the single-cell data (in the form of a graph) are processed separately, and the processed outputs are fused and further processed in the 'prediction head' to obtain classification probabilities. In position-aware SNOWFLAKE, the morphological information is blended into the single-cell graph through additional node features and edge features; this blended graph is processed (using GNN) and is used to obtain the classification results (graph pooling). (C) Pie graph showing the distribution of follicle database tissue distribution. (D) Pie graph and Sankey plot showing the distribution of NIH-COVID follicle database tissue distribution, COVID status distribution. 

# Organization

## Notebooks 
"notebooks" folder contains jupyter notebook script used.

## Source code
"src" folder contains customs scripts used.
