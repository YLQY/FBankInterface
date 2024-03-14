
#include "FBankInterface.h"
#include "Eigen/Core"

#pragma warning(disable:4996)

float* FBank(char* file_name)
{
    FILE* rf = fopen(file_name, "rb+");
    fseek(rf, 0, SEEK_END);
    float wav_dur = (ftell(rf) - 44) / (16000 * 2.0);
    fseek(rf, 44, SEEK_SET);
    int time_stride = 40;
    int read_len = time_stride * 2 * 16;
    char buffer[40 * 2 * 16] = { 0 };
    int read_count = 0;

    std::vector<float> feed_data;
    while ((read_count = fread(buffer, sizeof(char), read_len, rf)) > 0) {
        int feed_size = read_count / 2;
        const short* sdata = (const short*)buffer;
        for (int i = 0; i < feed_size; ++i) {
            feed_data.push_back(float(sdata[i] / 32768.0f));
        }
        memset(buffer, 0, read_len);
    }

    std::vector<std::vector<float>> feature_out;
    FeatureConfig config;
    auto feature_extract = std::make_shared<FeatureExtract>(config);
    feature_out = feature_extract->GetFeature(feed_data);
    fclose(rf);


    int rows = feature_out.size();
    int cols = feature_out[0].size();

    float* array1D = new float[rows * cols];


    std::size_t offset = 0;
    for (const auto& row : feature_out) {
        std::copy(row.begin(), row.end(), array1D + offset);
        offset += row.size();
    }

    return array1D;
}



