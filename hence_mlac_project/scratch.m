addpath('matlab');
num_symbols = 256;
sym = 10;
pix_norm = (double(sym) / (num_symbols-1)) * 2 - 1;
mu_k = pix_norm + [-0.1, 0, 0.1];
pi_k = [1/3, 1/3, 1/3];
log_s_k = [-2, -2, -2];
[pi_q, mu_q, log_s_q] = quantizeParameters(pi_k, mu_k, log_s_k);

% Test boundaries directly
boundaries = [(sym - 0.5), (sym + 0.5)];
boundaries_norm = (boundaries / (num_symbols - 1)) * 2 - 1;
cdf_vals = mixtureLogisticCDF(boundaries_norm, pi_q, mu_q, log_s_q);
fprintf('boundaries_norm = [%f, %f]\n', boundaries_norm(1), boundaries_norm(2));
fprintf('cdf_vals = [%f, %f]\n', cdf_vals(1), cdf_vals(2));
fprintf('diff = %f\n', cdf_vals(2) - cdf_vals(1));
fprintf('freq = %d\n', round( (cdf_vals(2) - cdf_vals(1)) * 65536 ));
