#pragma once
#ifndef FRONTEND_FEATURE_EXTRACT_H_
#define FRONTEND_FEATURE_EXTRACT_H_
#include <vector>
#include <complex>
#include <iostream>
#include "Eigen/Core"
#include "unsupported/Eigen/FFT"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define WHISPER_CHUNK_SIZE 30
typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorf;
typedef Eigen::Matrix<std::complex<float>, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorcf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrixf;
typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrixcf;


struct FeatureConfig {
	int sample_rate;
	int n_fft;
	int n_hop;
	std::string window;
	bool center;
	std::string pad_mode;
	float power;
	int n_mels;
	int fmin;
	int fmax;
	FeatureConfig(int sample_rate = 16000, int n_fft = 400, int n_hop = 160, std::string window = "hann",
		bool center = true, std::string pad_mode = "edge", float power = 2.0, int n_mels = 80,
		int fmin =0,int fmax=8000)
		:sample_rate(sample_rate),
		n_fft(n_fft),
		n_hop(n_hop),
		window(window),
		center(center),
		pad_mode(pad_mode),
		power(power),
		n_mels(n_mels),
		fmin(fmin),
		fmax(fmax){

	}
};

class FeatureExtract {
public:
	explicit FeatureExtract(const FeatureConfig& config);
	std::vector<std::vector<float>> GetFeature(std::vector<float>& x);
private:
	Matrixf MelSpectrogram(Vectorf& x);
	Vectorf Pad(Vectorf& x, int left, int right, float value);
	Matrixcf Stft(Vectorf& x);
	Matrixf MelFilter();
	Matrixf Spectrogram(Matrixcf& X);
	void RemoveRow(Matrixcf& matrix, size_t row_to_remove);
private:
	const FeatureConfig& config_;
};





#endif