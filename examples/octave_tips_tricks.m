% file for octave tips and tricks

% ---------------------------------------------
% VECTOR OF LABELS --> MATRIX OF FEATURE VECTORS
% ---------------------------------------------
% vector of labels, into a matrix of feature vectors in num_labels space
% given a vecor of numbers, assume they are labels, from 1 to num_labels
num_labels = 4;
vec = [1 3 4 4]'; % this is the same as vec = [1; 3; 4; 4];
% solution
matrix_features = [1:num_labels] == vec;
% -------
% output
% -------
% 1 0 0 0
% 0 0 1 0
% 0 0 0 1
% 0 0 0 1

% ---------------------------------------
% VECTOR OF LABELS --> VECTOR OF FEATURES
% ---------------------------------------
num_features = 5;
features_vec = zeros(num_features, 1);
indices = [1; 5; 4; 2; 1];
% solution
features_vec(indices) = 1;
% ------
% output
% ------
% 1 1 0 1 1
