//
// Created by 吴子凡 on 2020/7/8.
//

#include "img_process.h"



void pad2square(Mat &src,Mat &dst){
    /**
     * pad the img 2 square
     */
    int w = src.cols;
    int h = src.rows;
    int top,bottom,left,right;
    int diff;
    if(h>w){
        // pad 2 w
        diff = h-w;
        top=0,bottom=0;
        left=diff/2;
        right = diff-left;
    }
    else{
        // h<=w pad 2 h
        diff = w-h;
        left=0,right=0;
        top = diff/2;
        bottom = diff-top;
    }
    copyMakeBorder(src, dst, top, bottom, left, right, cv::BORDER_CONSTANT, 0);
}

void mypreprocess(Mat &src,Mat &dst,int current_dim){
    // bgr2rgb
    Mat rgb;
    cv::cvtColor(src, rgb,  cv::COLOR_BGR2RGB);
    // pad2square
    Mat square;
    pad2square(rgb,square);
    // resize
    Mat resized_img;
    cv::resize(square, resized_img, cv::Size(current_dim, current_dim), 0,0,cv::INTER_CUBIC);
    // normalization
    resized_img.convertTo(dst, CV_32F, 1.0/255);
}