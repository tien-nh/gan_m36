Step 1: Train expert (50 epochs)
Step 2: Train GAN (300 epochs)
Step 3: Generate training image into folder Generated_M36
Step 4: Compute atlas and save in disease_atlas.npy
Step 5: Train ViT (using Generated_M36, disease_atlas.npy, fold_3_val.csv to eval)
Step 6: Generate test image into folder Generated_Test_M36
Step 7: Eval at evaluate_model.py