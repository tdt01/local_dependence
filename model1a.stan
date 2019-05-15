// Model DEP

data {
	int N;					// Sample size
	int Nsub;                               // Number of subjects

	int K;					// Number of items
	int R;					// Number of latent factors
	int p;					// Number of covariates

	int ID[N];

	int cumu[Nsub];
	int repme[Nsub];

	int Y[N, K];
	int missing_ID[N, K];

	//vector[p] X[N];
	matrix[N, p] X;

	int ncate4;
	int ncate5;
	int ncate6;
	int ncate7;
}

parameters {

	real theta1;
	real theta2;
	real theta3;

	ordered[ncate4-1] theta4;
	ordered[ncate5-1] theta5;
	ordered[ncate6-1] theta6;
	ordered[ncate7-1] theta7;

	real mu_theta;
	real<lower=0.000001> sigma_theta;

	real<lower=0.000001> lambda11;
	real<lower=0.000001> lambda21;
	real<lower=0.000001> lambda31;
	real<lower=0.000001> lambda42;
	real<lower=0.000001> lambda52;
	real<lower=0.000001> lambda62;
	real<lower=0.000001> lambda72;


	real<lower=0.000001> sigma_lambda;

	matrix[K, p] beta;

	matrix[Nsub, K] b_raw;
	vector<lower=0.000001>[K] sigma_bk;

	vector[R] xi[N];
	
	matrix[R, R] Gamma; //Transition matrix

	real<lower=-1, upper=1> rho; // correlation coefficient
}

transformed parameters {

	matrix[K, R] lambda_matrix;

	matrix[Nsub, K] b;

	corr_matrix[R] Omega;

	cov_matrix[R] SIGMA;

	lambda_matrix[1, 1]=lambda11;
	lambda_matrix[2, 1]=lambda21;
	lambda_matrix[3, 1]=lambda31;

	lambda_matrix[4, 2]=lambda42;
	lambda_matrix[5, 2]=lambda52;
	lambda_matrix[6, 2]=lambda62;
	lambda_matrix[7, 2]=lambda72;

	lambda_matrix[1, 2]=0;
	lambda_matrix[2, 2]=0;
	lambda_matrix[3, 2]=0;

	lambda_matrix[4, 1]=0;
	lambda_matrix[5, 1]=0;
	lambda_matrix[6, 1]=0;
	lambda_matrix[7, 1]=0;





	for (i in 1 : Nsub){
		for (k in 1 : K){
			b[i, k] = b_raw[i, k] * sigma_bk[k];
		}
	}

	Omega = [[1, rho], [rho, 1]];

	SIGMA = Omega - Gamma * Omega * Gamma';
}

model{
	// Prior

	theta1 ~ normal(mu_theta, sigma_theta);
	theta2 ~ normal(mu_theta, sigma_theta);
	theta3 ~ normal(mu_theta, sigma_theta);

	theta4 ~ normal(mu_theta, sigma_theta);
	theta5 ~ normal(mu_theta, sigma_theta);
	theta6 ~ normal(mu_theta, sigma_theta);
	theta7 ~ normal(mu_theta, sigma_theta);

	mu_theta ~ cauchy(0, 5);
	sigma_theta ~ cauchy(0, 5);

	lambda11 ~ normal(1, sigma_lambda);
	lambda21 ~ normal(1, sigma_lambda);
	lambda31 ~ normal(1, sigma_lambda);
	lambda42 ~ normal(1, sigma_lambda);
	lambda52 ~ normal(1, sigma_lambda);
	lambda62 ~ normal(1, sigma_lambda);
	lambda72 ~ normal(1, sigma_lambda);


	sigma_lambda ~ cauchy(0, 5);

	to_vector(beta) ~ cauchy(0, 5);

	to_vector(Gamma) ~ normal(0, 10);

	sigma_bk ~ cauchy(0, 5);
	to_vector(b_raw) ~ normal(0, 1);

	// At time=1

	for (i in 1 : Nsub){
		
		int k;

		k = cumu[i] - repme[i] + 1;

		xi[k] ~ multi_normal([0, 0]', Omega);
	}

	// Now is time = 2 to end

	for (i in 1 : Nsub){
		for (j in 2 : repme[i]){

			int k;

			k = cumu[i] - repme[i] + j;
					
			xi[k] ~ multi_normal(Gamma * xi[k-1], SIGMA);
		}
	}

	// likelihood

	for (i in 1 : N){

		if (missing_ID[i, 1] == 0){Y[i, 1] ~ bernoulli_logit(theta1 + beta[1, 1] * X[i, 1] + beta[1, 2] * X[i, 2] + lambda_matrix[1, ] * xi[i] + b[ID[i], 1]);}
		if (missing_ID[i, 2] == 0){Y[i, 2] ~ bernoulli_logit(theta2 + beta[2, 1] * X[i, 1] + beta[2, 2] * X[i, 2] + lambda_matrix[2, ] * xi[i] + b[ID[i], 2]);}
		if (missing_ID[i, 3] == 0){Y[i, 3] ~ bernoulli_logit(theta3 + beta[3, 1] * X[i, 1] + beta[3, 2] * X[i, 2] + lambda_matrix[3, ] * xi[i] + b[ID[i], 3]);}

		if (missing_ID[i, 4] == 0){Y[i, 4] ~ ordered_logistic(beta[4, 1] * X[i, 1] + beta[4, 2] * X[i, 2] + lambda_matrix[4, ] * xi[i] + b[ID[i], 4], theta4);}
		if (missing_ID[i, 5] == 0){Y[i, 5] ~ ordered_logistic(beta[5, 1] * X[i, 1] + beta[5, 2] * X[i, 2] + lambda_matrix[5, ] * xi[i] + b[ID[i], 5], theta5);}
		if (missing_ID[i, 6] == 0){Y[i, 6] ~ ordered_logistic(beta[6, 1] * X[i, 1] + beta[6, 2] * X[i, 2] + lambda_matrix[6, ] * xi[i] + b[ID[i], 6], theta6);}
		if (missing_ID[i, 7] == 0){Y[i, 7] ~ ordered_logistic(beta[7, 1] * X[i, 1] + beta[7, 2] * X[i, 2] + lambda_matrix[7, ] * xi[i] + b[ID[i], 7], theta7);}
	}
}

