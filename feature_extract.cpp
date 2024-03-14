#include "feature_extract.h"

FeatureExtract::FeatureExtract(const FeatureConfig& config)
	:config_(config) {

}


std::vector<std::vector<float>> FeatureExtract::GetFeature(std::vector<float>& x) {
	std::vector<float> x_padded;
	int pad_size = WHISPER_CHUNK_SIZE * config_.sample_rate;
	if (x.size() < pad_size) {
		pad_size = pad_size - x.size();
		x_padded.resize(pad_size, 0);
		x.insert(x.end(), x_padded.begin(), x_padded.end());
	}
	else {
		//TODO
	}
	Vectorf map_x = Eigen::Map<Vectorf>(x.data(), x.size());
	Matrixf mel = MelSpectrogram(map_x);
	auto log_sp = mel.array().max(1e-10).log10();
	auto norm_log_sp = (log_sp.cwiseMax(log_sp.maxCoeff() - 8.0f) + 4.0) / 4.0;
	std::vector<std::vector<float>> log_mel_vector(norm_log_sp.rows(), std::vector<float>(norm_log_sp.cols(), 0.f));
	for (int i = 0; i < norm_log_sp.rows(); ++i) {
		auto& row = log_mel_vector[i];
		Eigen::Map<Vectorf>(row.data(), row.size()) = norm_log_sp.row(i);
	}
	return log_mel_vector;
}

Vectorf FeatureExtract::Pad(Vectorf& x, int left, int right, float value) {
	Vectorf x_paded = Vectorf::Constant(left + x.size() + right, value);
	x_paded.segment(left, x.size()) = x;
	if (config_.pad_mode.compare("reflect") == 0) {
		for (int i = 0; i < left; ++i) {
			x_paded[i] = x[left - i];
		}
		for (int i = left; i < left + right; ++i) {
			x_paded[i + x.size()] = x[x.size() - 2 - i + left];
		}
	}
	if (config_.pad_mode.compare("symmetric") == 0) {
		for (int i = 0; i < left; ++i) {
			x_paded[i] = x[left - i - 1];
		}
		for (int i = left; i < left + right; ++i) {
			x_paded[i + x.size()] = x[x.size() - 1 - i + left];
		}
	}

	if (config_.pad_mode.compare("edge") == 0) {
		for (int i = 0; i < left; ++i) {
			x_paded[i] = x[0];
		}
		for (int i = left; i < left + right; ++i) {
			x_paded[i + x.size()] = x[x.size() - 1];
		}
	}
	return x_paded;
}



Matrixcf FeatureExtract::Stft(Vectorf& x) {
	//hanning
	Vectorf window = 0.5 * (1.f - (Vectorf::LinSpaced(config_.n_fft, 0.f,
		static_cast<float>(config_.n_fft - 1)) * 2.f * M_PI / config_.n_fft).array().cos());
	int pad_len = config_.center ? config_.n_fft / 2 : 0;
	Vectorf x_paded = Pad(x, pad_len, pad_len, 0.f);
	int n_f = config_.n_fft / 2 + 1;
	int n_frames = 1 + (x_paded.size() - config_.n_fft) / config_.n_hop;
	Matrixcf X(n_frames, config_.n_fft);
	Eigen::FFT<float> fft;
	for (int i = 0; i < n_frames; ++i) {
		Vectorf x_frame = window.array() * x_paded.segment(i * config_.n_hop, config_.n_fft).array();
		X.row(i) = fft.fwd(x_frame);
	}
	return X.leftCols(n_f);
}




Matrixf FeatureExtract::MelFilter() {
	int n_f = config_.n_fft / 2 + 1;
	Vectorf fft_freqs = (Vectorf::LinSpaced(n_f, 0.f, static_cast<float>(n_f - 1)) * config_.sample_rate) / config_.n_fft;
	float f_min = 0.f;
	float f_sp = 200.f / 3.f;
	float min_log_hz = 1000.f;
	float min_log_mel = (min_log_hz - f_min) / f_sp;
	float logstep = logf(6.4f) / 27.f;
	auto hz_to_mel = [=](int hz, bool htk = false)->float {
		if (htk) {
			return 2595.0f * log10f(1.0f + hz / 700.0f);
		}
		float mel = (hz - f_min) / f_sp;
		if (hz >= min_log_hz) {
			mel = min_log_mel + logf(hz / min_log_hz) / logstep;
		}
		return mel;
	};
	auto mel_to_hz = [=](Vectorf& mels, bool htk = false) -> Vectorf {
		if (htk) {
			return 700.0f * (Vectorf::Constant(config_.n_mels + 2, 10.f).array().pow(mels.array() / 2595.0f) - 1.0f);
		}
		return (mels.array() > min_log_mel).select(((mels.array() - min_log_mel) * logstep).exp() * min_log_hz, (mels * f_sp).array() + f_min);
	};
	float min_mel = hz_to_mel(config_.fmin);
	float max_mel = hz_to_mel(config_.fmax);
	Vectorf mels = Vectorf::LinSpaced(config_.n_mels + 2, min_mel, max_mel);
	Vectorf mel_f = mel_to_hz(mels);
	Vectorf fdiff = mel_f.segment(1, mel_f.size() - 1) - mel_f.segment(0, mel_f.size() - 1);
	Matrixf ramps = mel_f.replicate(n_f, 1).transpose().array() - fft_freqs.replicate(config_.n_mels + 2, 1).array();
	Matrixf lower = -ramps.topRows(config_.n_mels).array() / fdiff.segment(0, config_.n_mels).transpose().replicate(1, n_f).array();
	Matrixf upper = ramps.bottomRows(config_.n_mels).array() / fdiff.segment(1, config_.n_mels).transpose().replicate(1, n_f).array();
	Matrixf weights = (lower.array() < upper.array()).select(lower, upper).cwiseMax(0);
	auto enorm = (2.0 / (mel_f.segment(2, config_.n_mels) - mel_f.segment(0, config_.n_mels)).array()).transpose().replicate(1, n_f);
	weights = weights.array() * enorm;
	return weights;
}

Matrixf FeatureExtract::Spectrogram(Matrixcf& X) {
	return X.cwiseAbs().array().pow(config_.power);
}

void FeatureExtract::RemoveRow(Matrixcf& matrix, size_t row_to_remove) {
	size_t num_rows = matrix.rows() - 1;
	size_t num_cols = matrix.cols();

	if (row_to_remove < num_rows) {
		matrix.block(row_to_remove, 0, num_rows - row_to_remove, num_cols) = matrix.block(row_to_remove + 1, 0, num_rows - row_to_remove, num_cols);
	}
	matrix.conservativeResize(num_rows, num_cols);
}

Matrixf FeatureExtract::MelSpectrogram(Vectorf& x) {
	Matrixcf X = Stft(x);
	size_t row_to_remove = X.rows() - 1;
	RemoveRow(X, row_to_remove);
	Matrixf mel_basis = MelFilter();
	Matrixf sp = Spectrogram(X);
	Matrixf mel = mel_basis * sp.transpose();
	return mel;
}




