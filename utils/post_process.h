//
// Created by 吴子凡 on 2020/7/13.
//

#ifndef TENSORRTTEST_POST_PROCESS_H
#define TENSORRTTEST_POST_PROCESS_H
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;


namespace post_process{
    typedef struct DetectionRes {
        float prob,label1,label2;
        float x1,y1,x2,y2,score;
        float max_label,max_id;
    } DetectionRes;
    float sigmoid(float in);
    float exponential(float in);
    float arg_max(vector<float> &label_list);
    void DoNms(vector<DetectionRes>& detections, float nmsThresh);
    DetectionRes rescale(DetectionRes box, int height,int width,int current_dim);
    float* merge(float* out1, float* out2,  int bsize_out1, int bsize_out2);



}





#endif //TENSORRTTEST_POST_PROCESS_H
