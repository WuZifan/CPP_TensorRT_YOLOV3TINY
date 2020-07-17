//
// Created by 吴子凡 on 2020/7/13.
//

#include "post_process.h"
namespace post_process{

float sigmoid(float in) {
    return 1.f / (1.f + exp(-in));
}

float exponential(float in) {
    return exp(in);
}


float arg_max(vector<float> &label_list){
    float ret_id = 0;
    float last_max = 0;
    for(int i=0;i<label_list.size();i++){
        if(label_list[i]>last_max){
            last_max=label_list[i];
            ret_id = i;
        }

    }
    return ret_id;
}

void DoNms(vector<DetectionRes>& detections, float nmsThresh){
    auto iouCompute = [](DetectionRes &lbox, DetectionRes &rbox) {
        float interBox[] = {
                max(lbox.x1, rbox.x1), //x1
                max(lbox.y1, rbox.y1), //y1
                min(lbox.x2, rbox.x2), //x2
                min(lbox.y2, rbox.y2), //y2
        };

        if (interBox[1] >= interBox[3] || interBox[0] >= interBox[2])
            return 0.0f;

        float interBoxS = (interBox[2] - interBox[0] + 1) * (interBox[3] - interBox[1] + 1);
        float lbox_area = (lbox.x2-lbox.x1+1)*(lbox.y2-lbox.y1+1);
        float rbox_area = (rbox.x2-rbox.x1+1)*(rbox.y2-rbox.y1+1);
        float extra = 0.00001;

        return interBoxS / (lbox_area+rbox_area-interBoxS+extra);
    };

    vector<DetectionRes> result;
    for (unsigned int m = 0; m < detections.size(); ++m) {
        result.push_back(detections[m]);
        for (unsigned int n = m + 1; n < detections.size(); ++n) {
            if (iouCompute(detections[m], detections[n]) > nmsThresh) {
                detections.erase(detections.begin() + n);
                --n;
            }
        }
    }
    detections = move(result);
}

DetectionRes rescale(DetectionRes box, int height,int width,int current_dim){
    float pad_x = max(height-width,0)*(1.0*current_dim/max(height,width));
    float pad_y = max(width-height,0)*(1.0*current_dim/max(height,width));

    float unpad_h = current_dim - pad_y;
    float unpad_w = current_dim - pad_x;

    box.x1 = ((box.x1-(pad_x/2))/unpad_w)*width;
    box.y1 = ((box.y1-(pad_y/2))/unpad_h)*height;
    box.x2 = ((box.x2-(pad_x/2))/unpad_w)*width;
    box.y2 = ((box.y2-(pad_y/2))/unpad_h)*height;

    return box;
}

float* merge(float* out1, float* out2,  int bsize_out1, int bsize_out2)
{
    /**
     * 合并两个输出
     */
    float* out_total = new float[bsize_out1 + bsize_out2];

    for (int j = 0; j < bsize_out1; ++j)
    {
        int index = j;
        out_total[index] = out1[j];
    }

    for (int j = 0; j < bsize_out2; ++j)
    {
        int index = j + bsize_out1;
        out_total[index] = out2[j];
    }

    return out_total;
}

}