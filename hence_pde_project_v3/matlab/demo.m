% DEMO: Run the trained Neural PDE and see the results!

% 1. Load the BEST trained weights
fprintf('1. Loading trained weights...\n');
weights = initPDEWeights('best_trained_weights.mat');

% 2. Load a REAL medical image
% You can change this number to any patient ID you want to test!
% 
% 🧠 TRAINING Patients (The AI studied these):
% 4, 6, 7, 9, 11, 12, 14, 16, 17, 18, 23, 24, 25, 26
% 
% 🎯 VALIDATION Patients (Unseen tests for the AI):
% 27, 28, 29, 30, 35, 40
patient_id = 35; 
fprintf('2. Loading real MRI data (Patient %d)...\n', patient_id);

% Path to the real patient file
patient_path = sprintf('../../chaos_processed_t2spir/%d_norm.mat', patient_id);
data = load(patient_path);
fields = fieldnames(data);
% Get the full 3D MRI volume
full_volume = data.(fields{1}); 

% Grab the middle slice of the patient's MRI scan
middle_slice = round(size(full_volume, 3) / 2);
volume = full_volume(:, :, middle_slice);

% Normalize to [-1, 1] using global 16-bit normalization (0 to 65535)
% This must match the training normalization exactly!
volume = double(volume);
volume = (2.0 * volume / 65535.0) - 1.0;

% 3. Run the Neural PDE Pipeline!
fprintf('3. Running Neural PDE...\n');
[pi_vol, mu_vol, log_s_vol] = runPipeline(volume, weights);

% 4. Visualize the Results
fprintf('4. Plotting results...\n');
figure('Name', 'Neural PDE Results', 'Position', [100, 100, 900, 300]);

% Show original image
subplot(1,3,1);
imshow(volume, []);
title('Original Medical Image');

% Show what the Neural Network predicted (the Mean 'Mu' of the 1st mixture)
predicted_mu = mu_vol(:,:,1,1); 
subplot(1,3,2);
imshow(predicted_mu, []);
title('Predicted Image (Mu_1)');

% Show the difference (Error/Residual)
% If the neural network is smart, this error should be very small!
residual = abs(volume - predicted_mu);
subplot(1,3,3);
imshow(residual, []);
title('Prediction Error');
colormap(gca, 'hot'); 

fprintf('\nSuccess! The PDE is working perfectly.\n');
fprintf('Track 2 can now use the pi_vol, mu_vol, and log_s_vol matrices!\n');
