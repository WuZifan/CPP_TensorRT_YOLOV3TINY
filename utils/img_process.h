//
// Created by 吴子凡 on 2020/7/8.
//

#ifndef TENSORRTTEST_IMG_PROCESS_H
#define TENSORRTTEST_IMG_PROCESS_H
#include <opencv2/opencv.hpp>

using namespace cv;

void pad2square(Mat &src,Mat &dst);
void mypreprocess(Mat &src,Mat &dst,int current_dim);


#endif //TENSORRTTEST_IMG_PROCESS_H
